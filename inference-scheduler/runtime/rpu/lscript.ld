/*
 * Copyright 2026 Ivan Gubochkin
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * lscript.ld — the inference firmware on the Cortex-R5 (RPU_PLAN track B1).
 *
 * RPU core 0's own TCMs: 64 KiB of ATCM at 0x00000000 (the vector table,
 * code, read-only weights) and 64 KiB of BTCM at 0x00020000 (the control
 * block the A53 reads through /dev/mem at 0xFFE20000, data, bss, heap,
 * stack) — the same addresses in split and in lockstep mode (lockstep adds
 * core 1's banks above each; raise the lengths to 0x20000 for a bigger
 * firmware).  The DMA pool is DDR memory the A53 allocates (the region the
 * control block names), not a section here.
 */
_STACK_SIZE = DEFINED(_STACK_SIZE) ? _STACK_SIZE : 0x2000;
_HEAP_SIZE  = DEFINED(_HEAP_SIZE)  ? _HEAP_SIZE  : 0x4000;
_ABORT_STACK_SIZE = DEFINED(_ABORT_STACK_SIZE) ? _ABORT_STACK_SIZE : 1024;
_SUPERVISOR_STACK_SIZE = DEFINED(_SUPERVISOR_STACK_SIZE) ? _SUPERVISOR_STACK_SIZE : 2048;
_IRQ_STACK_SIZE = DEFINED(_IRQ_STACK_SIZE) ? _IRQ_STACK_SIZE : 1024;
_FIQ_STACK_SIZE = DEFINED(_FIQ_STACK_SIZE) ? _FIQ_STACK_SIZE : 1024;
_UNDEF_STACK_SIZE = DEFINED(_UNDEF_STACK_SIZE) ? _UNDEF_STACK_SIZE : 1024;

/* The ELF's addresses are the R5's own (the TCMs at 0x0 / 0x20000): the xlnx
 * remoteproc driver maps the banks' global aperture (0xFFE00000 / 0xFFE20000)
 * onto these device addresses when it loads the firmware. */
MEMORY
{
   psu_r5_tcm_ram_a : ORIGIN = 0x00000000, LENGTH = 0x00010000
   psu_r5_tcm_ram_b : ORIGIN = 0x00020000, LENGTH = 0x00010000
}

ENTRY(_vector_table)

SECTIONS
{
.vectors : {
   KEEP (*(.vectors))
   *(.boot)
} > psu_r5_tcm_ram_a

.text : {
   *(.text)
   *(.text.*)
   *(.gnu.linkonce.t.*)
   *(.plt)
   *(.gnu_warning)
   *(.gcc_execpt_table)
   *(.glue_7)
   *(.glue_7t)
   *(.ARM.extab)
   *(.gnu.linkonce.armextab.*)
} > psu_r5_tcm_ram_a

.init : { KEEP (*(.init)) } > psu_r5_tcm_ram_a
.fini : { KEEP (*(.fini)) } > psu_r5_tcm_ram_a

.rodata : {
   __rodata_start = .;
   *(.rodata)
   *(.rodata.*)
   *(.gnu.linkonce.r.*)
   __rodata_end = .;
} > psu_r5_tcm_ram_a

.rodata1 : {
   __rodata1_start = .;
   *(.rodata1)
   *(.rodata1.*)
   __rodata1_end = .;
} > psu_r5_tcm_ram_a

.ARM.exidx : {
   __exidx_start = .;
   *(.ARM.exidx*)
   *(.gnu.linkonce.armexidix.*.*)
   __exidx_end = .;
} > psu_r5_tcm_ram_a

.eh_frame : { *(.eh_frame) } > psu_r5_tcm_ram_a

.note.gnu.build-id : { *(.note.gnu.build-id) } > psu_r5_tcm_ram_a

.init_array : {
   __init_array_start = .;
   KEEP (*(SORT(.init_array.*)))
   KEEP (*(.init_array))
   __init_array_end = .;
} > psu_r5_tcm_ram_a

.fini_array : {
   __fini_array_start = .;
   KEEP (*(SORT(.fini_array.*)))
   KEEP (*(.fini_array))
   __fini_array_end = .;
} > psu_r5_tcm_ram_a

/* The control block first in the BTCM: the A53 finds it at 0xFFE20000. */
.rpu_ctrl 0x00020000 : {
   KEEP (*(.rpu_ctrl))
} > psu_r5_tcm_ram_b

.data : {
   __data_start = .;
   *(.data)
   *(.data.*)
   *(.gnu.linkonce.d.*)
   *(.jcr)
   *(.got)
   *(.got.plt)
   __data_end = .;
} > psu_r5_tcm_ram_b

.data1 : {
   __data1_start = .;
   *(.data1)
   *(.data1.*)
   __data1_end = .;
} > psu_r5_tcm_ram_b

.ctors : {
   __CTOR_LIST__ = .;
   ___CTORS_LIST___ = .;
   KEEP (*crtbegin.o(.ctors))
   KEEP (*(EXCLUDE_FILE(*crtend.o) .ctors))
   KEEP (*(SORT(.ctors.*)))
   KEEP (*(.ctors))
   __CTOR_END__ = .;
   ___CTORS_END___ = .;
} > psu_r5_tcm_ram_b

.dtors : {
   __DTOR_LIST__ = .;
   ___DTORS_LIST___ = .;
   KEEP (*crtbegin.o(.dtors))
   KEEP (*(EXCLUDE_FILE(*crtend.o) .dtors))
   KEEP (*(SORT(.dtors.*)))
   KEEP (*(.dtors))
   __DTOR_END__ = .;
   ___DTORS_END___ = .;
} > psu_r5_tcm_ram_b

.sdata : {
   __sdata_start = .;
   *(.sdata)
   *(.sdata.*)
   *(.gnu.linkonce.s.*)
   __sdata_end = .;
} > psu_r5_tcm_ram_b

.sbss (NOLOAD) : {
   __sbss_start = .;
   *(.sbss)
   *(.sbss.*)
   *(.gnu.linkonce.sb.*)
   __sbss_end = .;
} > psu_r5_tcm_ram_b

.tdata : {
   __tdata_start = .;
   *(.tdata)
   *(.tdata.*)
   *(.gnu.linkonce.td.*)
   __tdata_end = .;
} > psu_r5_tcm_ram_b

.tbss : {
   __tbss_start = .;
   *(.tbss)
   *(.tbss.*)
   *(.gnu.linkonce.tb.*)
   __tbss_end = .;
} > psu_r5_tcm_ram_b

.bss (NOLOAD) : {
   . = ALIGN(4);
   __bss_start__ = .;
   *(.bss)
   *(.bss.*)
   *(.gnu.linkonce.b.*)
   *(COMMON)
   . = ALIGN(4);
   __bss_end__ = .;
} > psu_r5_tcm_ram_b

_SDA_BASE_ = __sdata_start + ((__sbss_end - __sdata_start) / 2 );

.heap (NOLOAD) : {
   . = ALIGN(16);
   _heap = .;
   HeapBase = .;
   _heap_start = .;
   . += _HEAP_SIZE;
   _heap_end = .;
   HeapLimit = .;
} > psu_r5_tcm_ram_b

.stack (NOLOAD) : {
   . = ALIGN(16);
   _stack_end = .;
   . += _STACK_SIZE;
   . = ALIGN(16);
   _stack = .;
   __stack = _stack;
   . = ALIGN(16);
   _irq_stack_end = .;
   . += _IRQ_STACK_SIZE;
   . = ALIGN(16);
   __irq_stack = .;
   _supervisor_stack_end = .;
   . += _SUPERVISOR_STACK_SIZE;
   . = ALIGN(16);
   __supervisor_stack = .;
   _abort_stack_end = .;
   . += _ABORT_STACK_SIZE;
   . = ALIGN(16);
   __abort_stack = .;
   _fiq_stack_end = .;
   . += _FIQ_STACK_SIZE;
   . = ALIGN(16);
   __fiq_stack = .;
   _undef_stack_end = .;
   . += _UNDEF_STACK_SIZE;
   . = ALIGN(16);
   __undef_stack = .;
} > psu_r5_tcm_ram_b

_end = .;
}
