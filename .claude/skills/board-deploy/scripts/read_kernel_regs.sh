#!/usr/bin/env bash
# read_kernel_regs.sh [host] — PS AXI slave port widths + kernel ctrl/arg registers on the KV260.
host=${1:-192.168.100.8}
ssh -i ~/.ssh/kv260-testkey -o BatchMode=yes -o ConnectTimeout=10 root@"$host" 'python3 - <<EOF
import mmap, struct, os
fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
def rd(addr):
    base = addr & ~0xFFF; m = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
    v = struct.unpack("<I", m[addr-base:addr-base+4])[0]; m.close(); return v
print("PS slave port width fields (0=32b 1=64b 2=128b):")
for a, n in [(0xFD360000,"HPC0 rd"),(0xFD360014,"HPC0 wr"),(0xFD370000,"HPC1 rd"),(0xFD370014,"HPC1 wr"),(0xFD380000,"HP0 rd"),(0xFD380014,"HP0 wr")]:
    print("  %-8s %d" % (n, rd(a) & 3))
print("kernel control (bit0 start, bit1 done, bit2 idle):")
for b, n in [(0xA0000000,"vectorop"),(0xA0010000,"matmul"),(0xA0020000,"conv"),(0xA0030000,"pool")]:
    c = rd(b); print("  %-9s ctrl=0x%02x start=%d done=%d idle=%d" % (n, c, c&1, (c>>1)&1, (c>>2)&1))
regs = [("batch",0x40),("in_ch",0x48),("in_h",0x50),("in_w",0x58),("out_ch",0x60),("out_h",0x68),("out_w",0x70),("kh",0x78),("kw",0x80),("stride_h",0x88),("stride_w",0x90),("dil_h",0x98),("dil_w",0xa0),("pad_top",0xa8),("pad_left",0xb0),("has_bias",0xb8),("is_dw",0xc0)]
print("conv args:", " ".join("%s=%d" % (k, rd(0xA0020000 + o)) for k, o in regs))
EOF'
