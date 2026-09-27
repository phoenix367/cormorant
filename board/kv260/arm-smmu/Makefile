# On the board: build arm_smmu.ko against the running kernel's headers.
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	rm -f *.dtbo cormorant-smmu.dts

# On the host (needs dtc): the device-tree overlays.
#   smmu-enable.dtbo     enables &smmu (status okay, stream-match-mask 0x7f)
#   cormorant-smmu.dtbo  dts/kv260/cormorant.dts with zocl behind the SMMU
DTC ?= dtc
CORMORANT_DTS := ../../../dts/kv260/cormorant.dts

overlays: smmu-enable.dtbo cormorant-smmu.dtbo

cormorant-smmu.dts: $(CORMORANT_DTS)
	sed 's|compatible = "xlnx,zocl";|compatible = "xlnx,zocl";\n\t\t\t\tiommus = <\&smmu 0x200>;|' $< > $@

%.dtbo: %.dts
	$(DTC) -@ -I dts -O dtb -o $@ $<

.PHONY: all clean overlays
