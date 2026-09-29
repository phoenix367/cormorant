// SPDX-License-Identifier: GPL-2.0-only
/*
 * fpga_smmu_mem — DMA buffers for the KV260 PL kernels behind the ZynqMP SMMU.
 *
 * The PL kernels stream every tensor from one base address, so a buffer must
 * be contiguous in the address space they see.  Behind the SMMU that is the
 * IO virtual address space: this driver allocates ordinary pages (no CMA),
 * maps them with the DMA API — iommu-dma places them at one contiguous IOVA
 * range — and hands out that device address.  One buffer per open file:
 *
 *   FSM_IOC_ALLOC  pages in chunks of up to 2 MiB (zeroed), dma_map_sgtable,
 *                  checked to be one contiguous IOVA range
 *   mmap           cacheable CPU mapping (write-combined with FSM_ALLOC_WC)
 *   FSM_IOC_SYNC   cache maintenance on a byte range, split at the physically
 *                  contiguous chunks: iommu-dma syncs a range by translating
 *                  its start address only, so a range must not cross a chunk
 *   release        unmap and free
 *
 * The device must sit behind an IOMMU (device tree `iommus`): probe refuses
 * otherwise, since without translation the scattered pages would reach the
 * kernels as one bogus physical range.
 *
 * IOVA window: the kernels' AXI address map (Vivado address editor) forwards
 * only DDR_LOW (0–2 GiB), QSPI and DDR_HIGH (32–64 GiB) to the PS; anything
 * else gets DECERR from the interconnect and never reaches the SMMU.  A
 * 36-bit DMA mask makes iommu-dma allocate top-down from 64 GiB, i.e. in
 * DDR_HIGH — 32 GiB of IOVA space, room for size-aligned buffers of up to
 * FSM_MAX_SIZE — and every buffer is checked to lie in
 * [FSM_IOVA_BASE, FSM_IOVA_END).  (DDR_LOW would do for small buffers only:
 * a buffer over 1 GiB needs a 2 GiB-aligned slot, and IOVA 0 is reserved.)
 */
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "fpga_smmu_mem.h"

#define FSM_MAX_CHUNK_ORDER   9               /* 2 MiB with 4 KiB pages */
#define FSM_MAX_SIZE          (3ULL << 30)    /* per buffer */
#define FSM_IOVA_BITS         36              /* the mask's limit is FSM_IOVA_END */

static_assert(FSM_IOVA_END == 1ULL << FSM_IOVA_BITS, "IOVA window vs DMA mask");

struct fsm_dev {
	struct device     *dev;
	struct miscdevice  misc;
};

struct fsm_buf {
	struct fsm_dev    *fdev;
	struct mutex       lock;
	struct page      **pages;                 /* order-0, npages of them */
	unsigned long      npages;
	size_t             size;                  /* bytes, page multiple */
	struct sg_table    sgt;                   /* CPU view: physically contiguous chunks */
	dma_addr_t         iova;
	bool               allocated;
	bool               wc;
};

static struct fsm_dev *fsm_the_dev;           /* one instance */

static void fsm_free_pages(struct fsm_buf *b, unsigned long n)
{
	unsigned long i;

	for (i = 0; i < n; i++)
		__free_page(b->pages[i]);
}

/* Fill b->pages with npages zeroed order-0 pages, taken from the largest
 * chunks the allocator gives (fewer, longer physically contiguous runs). */
static int fsm_alloc_pages(struct fsm_buf *b)
{
	unsigned long done = 0;
	unsigned int order = FSM_MAX_CHUNK_ORDER;

	while (done < b->npages) {
		unsigned long left = b->npages - done, k;
		struct page *p;

		while (order && (1UL << order) > left)
			order--;
		for (;;) {
			/* Large chunks only when they come cheap; single pages
			 * may reclaim, but a request the board cannot hold fails
			 * with -ENOMEM rather than waking the OOM killer (no
			 * swap: it would kill whatever holds the most memory). */
			gfp_t gfp = GFP_KERNEL | __GFP_ZERO | __GFP_NOWARN;

			gfp |= order ? __GFP_NORETRY : __GFP_RETRY_MAYFAIL;
			p = alloc_pages(gfp, order);
			if (p || !order)
				break;
			order--;
		}
		if (!p) {
			fsm_free_pages(b, done);
			return -ENOMEM;
		}
		if (order)
			split_page(p, order);
		for (k = 0; k < (1UL << order); k++)
			b->pages[done + k] = p + k;
		done += 1UL << order;
		cond_resched();
	}
	return 0;
}

static long fsm_ioctl_alloc(struct fsm_buf *b, void __user *arg)
{
	struct device *dev = b->fdev->dev;
	struct fsm_alloc req;
	struct scatterlist *sg;
	dma_addr_t expect;
	unsigned int i;
	int ret;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (!req.size || req.size > FSM_MAX_SIZE || (req.flags & ~FSM_ALLOC_WC))
		return -EINVAL;

	mutex_lock(&b->lock);
	if (b->allocated) {
		ret = -EBUSY;
		goto out;
	}
	b->size = PAGE_ALIGN(req.size);
	b->npages = b->size >> PAGE_SHIFT;
	b->wc = !!(req.flags & FSM_ALLOC_WC);
	b->pages = kvmalloc_array(b->npages, sizeof(*b->pages), GFP_KERNEL);
	if (!b->pages) {
		ret = -ENOMEM;
		goto out;
	}
	ret = fsm_alloc_pages(b);
	if (ret)
		goto err_array;
	ret = sg_alloc_table_from_pages(&b->sgt, b->pages, b->npages, 0, b->size, GFP_KERNEL);
	if (ret)
		goto err_pages;
	/* Map for the device; for a non-coherent device this also cleans the
	 * (zeroed) pages out of the CPU caches. */
	ret = dma_map_sgtable(dev, &b->sgt, DMA_BIDIRECTIONAL, 0);
	if (ret)
		goto err_table;

	/* The kernels need one contiguous device range: check it. */
	b->iova = sg_dma_address(b->sgt.sgl);
	expect = b->iova;
	for_each_sgtable_dma_sg(&b->sgt, sg, i) {
		if (sg_dma_address(sg) != expect) {
			dev_err(dev, "buffer of %zu bytes is not contiguous in IOVA space\n", b->size);
			ret = -EFAULT;
			goto err_unmap;
		}
		expect += sg_dma_len(sg);
	}
	if (expect - b->iova < b->size) {
		ret = -EFAULT;
		goto err_unmap;
	}
	if (b->iova < FSM_IOVA_BASE || b->iova + b->size > FSM_IOVA_END) {
		dev_err(dev, "buffer of %zu bytes at IOVA %pad is outside the PL window\n",
			b->size, &b->iova);
		ret = -ENOSPC;
		goto err_unmap;
	}

	b->allocated = true;
	req.iova = b->iova;
	mutex_unlock(&b->lock);
	if (copy_to_user(arg, &req, sizeof(req)))
		return -EFAULT;
	return 0;

err_unmap:
	dma_unmap_sgtable(dev, &b->sgt, DMA_BIDIRECTIONAL, 0);
err_table:
	sg_free_table(&b->sgt);
err_pages:
	fsm_free_pages(b, b->npages);
err_array:
	kvfree(b->pages);
	b->pages = NULL;
out:
	mutex_unlock(&b->lock);
	return ret;
}

static long fsm_ioctl_sync(struct fsm_buf *b, void __user *arg)
{
	struct device *dev = b->fdev->dev;
	struct fsm_sync req;
	struct scatterlist *sg;
	u64 pos = 0, end;
	unsigned int i;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (req.dir != FSM_SYNC_TO_DEVICE && req.dir != FSM_SYNC_FROM_DEVICE)
		return -EINVAL;
	if (!b->allocated)
		return -EINVAL;
	if (req.offset > b->size || req.size > b->size - req.offset)
		return -EINVAL;
	if (b->wc || !req.size)
		return 0;                         /* write-combined: nothing cached */

	end = req.offset + req.size;
	/* Walk the CPU scatterlist: each entry is physically contiguous and
	 * sits at iova + pos in the (contiguous) device range. */
	for_each_sgtable_sg(&b->sgt, sg, i) {
		u64 s = max_t(u64, pos, req.offset);
		u64 e = min_t(u64, pos + sg->length, end);

		if (s < e) {
			if (req.dir == FSM_SYNC_TO_DEVICE)
				dma_sync_single_for_device(dev, b->iova + s, e - s, DMA_TO_DEVICE);
			else
				dma_sync_single_for_cpu(dev, b->iova + s, e - s, DMA_FROM_DEVICE);
		}
		pos += sg->length;
		if (pos >= end)
			break;
	}
	return 0;
}

static long fsm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct fsm_buf *b = filp->private_data;

	switch (cmd) {
	case FSM_IOC_ALLOC:
		return fsm_ioctl_alloc(b, (void __user *)arg);
	case FSM_IOC_SYNC:
		return fsm_ioctl_sync(b, (void __user *)arg);
	default:
		return -ENOTTY;
	}
}

static int fsm_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct fsm_buf *b = filp->private_data;
	unsigned long len = vma->vm_end - vma->vm_start;
	unsigned long num;
	int ret;

	if (!b->allocated)
		return -EINVAL;
	if (vma->vm_pgoff || len > b->size || !(vma->vm_flags & VM_SHARED))
		return -EINVAL;
	vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
	if (b->wc)
		vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	num = len >> PAGE_SHIFT;
	ret = vm_insert_pages(vma, vma->vm_start, b->pages, &num);
	return ret;
}

static int fsm_open(struct inode *inode, struct file *filp)
{
	struct fsm_buf *b = kzalloc(sizeof(*b), GFP_KERNEL);

	if (!b)
		return -ENOMEM;
	b->fdev = fsm_the_dev;
	mutex_init(&b->lock);
	filp->private_data = b;
	return 0;
}

static int fsm_release(struct inode *inode, struct file *filp)
{
	struct fsm_buf *b = filp->private_data;

	if (b->allocated) {
		dma_unmap_sgtable(b->fdev->dev, &b->sgt, DMA_BIDIRECTIONAL, 0);
		sg_free_table(&b->sgt);
		fsm_free_pages(b, b->npages);
		kvfree(b->pages);
	}
	kfree(b);
	return 0;
}

static const struct file_operations fsm_fops = {
	.owner          = THIS_MODULE,
	.open           = fsm_open,
	.release        = fsm_release,
	.unlocked_ioctl = fsm_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
	.mmap           = fsm_mmap,
};

static int fsm_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct iommu_domain *dom = iommu_get_domain_for_dev(dev);
	struct fsm_dev *fdev;
	int ret;

	if (!dom) {
		dev_err(dev, "not behind an IOMMU (iommus property / SMMU driver): refusing\n");
		return -ENODEV;
	}
	if (fsm_the_dev)
		return -EBUSY;
	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(FSM_IOVA_BITS));
	if (ret)
		return ret;
	dma_set_max_seg_size(dev, UINT_MAX);

	fdev = devm_kzalloc(dev, sizeof(*fdev), GFP_KERNEL);
	if (!fdev)
		return -ENOMEM;
	fdev->dev = dev;
	fdev->misc.minor = MISC_DYNAMIC_MINOR;
	fdev->misc.name = "fpga_smmu_mem";
	fdev->misc.fops = &fsm_fops;
	fdev->misc.mode = 0600;
	fsm_the_dev = fdev;
	ret = misc_register(&fdev->misc);
	if (ret) {
		fsm_the_dev = NULL;
		return ret;
	}
	platform_set_drvdata(pdev, fdev);
	dev_info(dev, "/dev/fpga_smmu_mem: IOMMU domain type %u, IOVA window 0x%llx-0x%llx\n",
		 dom->type, FSM_IOVA_BASE, FSM_IOVA_END - 1);
	return 0;
}

static int fsm_remove(struct platform_device *pdev)
{
	struct fsm_dev *fdev = platform_get_drvdata(pdev);

	misc_deregister(&fdev->misc);
	fsm_the_dev = NULL;
	return 0;
}

static const struct of_device_id fsm_of_match[] = {
	{ .compatible = "axi-demo,fpga-smmu-mem" },
	{ }
};
MODULE_DEVICE_TABLE(of, fsm_of_match);

static struct platform_driver fsm_driver = {
	.probe  = fsm_probe,
	.remove = fsm_remove,
	.driver = {
		.name           = "fpga_smmu_mem",
		.of_match_table = fsm_of_match,
	},
};
module_platform_driver(fsm_driver);

MODULE_DESCRIPTION("DMA buffers for PL kernels behind the ZynqMP SMMU");
MODULE_LICENSE("GPL");
