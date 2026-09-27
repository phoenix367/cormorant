# On the board: build fpga_smmu_mem.ko against the running kernel's headers.
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	rm -f *.dtbo

# On the host (needs dtc): the overlay.
DTC ?= dtc
overlay: smmu-mem.dtbo

%.dtbo: %.dts
	$(DTC) -@ -I dts -O dtb -o $@ $<

.PHONY: all clean overlay
