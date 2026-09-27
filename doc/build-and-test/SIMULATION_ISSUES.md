# MPSoC PS VIP Simulation — API Internals and Known Issues

Vivado 2025.2 / `zynq_ultra_ps_e_vip_v1_0` (module prefix `zynq_ultra_ps_e_vip_v1_0_22_`) / xsim

The testbenches this applies to are the four `cormorant_test_stand` benches
driven by `make behavior_test_<k>` (see §8).

---

## 1. PS VIP Memory Write API

### `PS.write_mem`

```systemverilog
`PS.write_mem(buf_mem, base_addr, nbytes);
```

| Parameter   | Type                    | Notes |
|-------------|-------------------------|-------|
| `buf_mem`   | `logic [CHUNK_BITS-1:0]`| Source buffer. `CHUNK_BITS = CHUNK_SIZE * 8 = 8192` in the test-stand benches. |
| `base_addr` | `[39:0]`                | Byte address in PS DDR space. Must be 4-byte aligned (otherwise the VIP prints an error and `$stop`s under `set_stop_on_error(1)`). |
| `nbytes`    | `int unsigned`          | Number of bytes to transfer. **Must not exceed the bytes actually filled in `buf_mem` (`CHUNK_SIZE`)**. |

`write_mem` is a backdoor load: it goes straight to `ddrc.ddr.write_mem()` (the sparse memory model, §7.6) with all strobes set, without any AXI transaction.

**Hard limit per call: the bench buffer, `CHUNK_SIZE = 1024` bytes.** The VIP's own `data` argument is `max_burst_bits` = 32768 bits wide (4096 bytes, §7.3), so the VIP accepts any `nbytes` up to 4096 without complaint.

Calling `write_mem` with `nbytes > CHUNK_SIZE` does NOT split the transfer. The VIP copies `nbytes` bytes of its 32768-bit argument, so the bytes past what the bench filled carry whatever the upper bits hold (`X` for a wider, partially filled buffer). The simulation continues without an immediate error; the `X` propagates into AXI read data later, causing a fatal `AXI4_ERRS_RDATA_X` assertion at the point the kernel reads those addresses.

**Correct pattern for large buffers — chunk loop:**

```systemverilog
localparam integer CHUNK_SIZE = 1024;
localparam integer CHUNK_BITS = CHUNK_SIZE * 8;

logic [CHUNK_BITS-1:0] buf_mem;
int unsigned n_chunks, rem;
n_chunks = nbytes / CHUNK_SIZE;
rem      = nbytes % CHUNK_SIZE;
for (int c = 0; c < n_chunks; c++)
    `PS.write_mem(buf_mem, base + 40'(c * CHUNK_SIZE), CHUNK_SIZE);
if (rem > 0)
    `PS.write_mem(buf_mem, base + 40'(n_chunks * CHUNK_SIZE), rem);
```

For ramp/varying patterns the inner fill loop must also be chunk-relative:

```systemverilog
localparam integer ELEM_BYTES = 2;  // ap_fixed<16,8>
int unsigned chunk_elems = CHUNK_SIZE / ELEM_BYTES;  // 512
for (int c = 0; c <= int'(n_chunks); c++) begin
    this_bytes = (c < int'(n_chunks)) ? CHUNK_SIZE : rem;
    if (this_bytes == 0) break;
    this_elems = this_bytes / ELEM_BYTES;
    for (int e = 0; e < int'(this_elems); e++) begin
        eidx = unsigned'(c) * chunk_elems + unsigned'(e);
        buf_mem[e*16 +: 16] = (eidx < n_elem) ? (val + 16'(eidx) * step) : 16'h0;
    end
    `PS.write_mem(buf_mem, base + 40'(c * CHUNK_SIZE), this_bytes);
end
```

**Symptom of exceeding the limit:** `AXI4_ERRS_RDATA_X` fatal from the AXI VIP protocol checker, appearing when the kernel issues read bursts that cover the affected addresses. Elements at indices >= 512 (byte offset >= 1024) are affected.

---

## 2. DDR Memory Array Addressing

The PS VIP internal DDR model is split into two sparse 1 GB banks (2^28 32-bit words each):

```
dut.<ps_inst>.inst.ddrc.ddr.ddr_mem0[index]   // word_addr[28] == 0  (byte 0x0000_0000–0x3FFF_FFFF)
dut.<ps_inst>.inst.ddrc.ddr.ddr_mem1[index]   // word_addr[28] == 1  (byte 0x4000_0000–0x7FFF_FFFF)
```

Word-address bits above 28 are ignored by the indexing, so higher byte addresses alias onto these two banks.

Each array element is a **32-bit word** (4 bytes). Index computation from a byte address:

```systemverilog
logic [39:0] ba;          // 40-bit byte address
logic [31:0] wa;          // 32-bit word address
int unsigned boff;        // byte offset within word (0-3)

wa   = ba[33:2];          // == byte_addr >> 2 for addresses < 4 GB
boff = int'(ba[1:0]);

if (wa[28] == 1'b0)
    target = ddr_mem0[wa[27:0]];
else
    target = ddr_mem1[wa[27:0]];
```

Direct array access bypasses the DDRC model entirely — useful for testbench monitors and workarounds that need to observe or inject data without going through the AXI stack.

---

## 3. `zynq_ultra_ps_e_vip_v1_0_22_arb_wr_6` Race Condition

### Root Cause

In `arb_wr_6`, each `if(prt_dvN) begin...end` dispatch block follows this order:

```systemverilog
if (prt_dv0) begin
    prt_req   = 1;                   // ← fires in active event region
    #0; prt_data = ...;              // ← settles in inactive region
    #0; prt_strb = ...;
    prt_addr  = ...;
    prt_bytes = ...;
end
```

`prt_req = 1` fires in the **active event region** of `posedge sw_clk`. The DDRC's `always @(posedge sw_clk)` block also fires in the same active region. Depending on elaboration/evaluation order, DDRC may sample `wr_req` high while `wr_addr`, `wr_bytes`, `wr_strb`, and `wr_data` still hold the **previous burst's values** (the `#0`-delayed assignments have not yet executed).

The result is that narrow writes — any beat whose WSTRB is not all ones — are silently written to the wrong address or dropped, and the racing write can also deposit stale data in the bytes such a beat leaves *unstrobed*.

### Effect on Simulation

- Originally found with 32-bit ports: the last element of an odd-count `ap_fixed<16,8>` vector (last beat WSTRB = `0x3`) was lost; even counts passed.
- With the current 128-bit ports, partial-strobe beats come from the ConvKernel and PoolingKernel `y` ports (byte-strobed run edges, `doc/kernels/CONV_OPTIMISATION.md` §2.38 / `doc/kernels/POOL_OPTIMISATION.md` §2.14) and from the MatmulKernel `c` writes (a 16-bit port — HLS does not widen it — so every beat is a partial strobe once upsized to the 128-bit PS port). VectorOPKernel writes every output word whole (tail lanes = 0), so it no longer produces partial beats.
- Stale-byte variant (`doc/kernels/CONV_OPTIMISATION.md` §2.38): a `y` beat with `WSTRB = 0x00ff` landed the line's pre-poison contents (a previous case's outputs at the same DDR line) in bytes 8..15. Single-case reruns hide it — it needs an earlier AXI write to the same line, so run the whole suite.
- In all cases the kernel's beat is correct; the DDR model holds the wrong bytes.

### Fix Option A — Patch `arb_wr_6` (VIP source edit)

Move `prt_req = 1` to **after** `prt_bytes` in every dispatch block. With `prt_req` last, it fires in the inactive event region (after all `#0` suspensions resolve), so DDRC sees `wr_req=0` on the current posedge and samples the fully-settled parameters at the **next** posedge.

A Python script (outside this repository) applies this to the Xilinx installed file:
`/home/ivan/vivado_projects/fix_arb_wr6_prt_req.py`.  The installed VIP source is currently unpatched; the test-stand benches rely on Option B instead.

Target file:
`/mnt/data/xilinx/2025.2/data/ip/xilinx/zynq_ultra_ps_e_vip_v1_0/hdl/zynq_ultra_ps_e_vip_v1_0_vl_rfs.sv`

Affects 36 occurrences across `wait_req` + `serv_req0..serv_req5`.

### Fix Option B — Testbench `ddrc_wr_fix` Workaround (no VIP edit)

Add a module-level `initial begin : ddrc_wr_fix` block that intercepts every `wr_req` rising edge, waits 1 ns (past all `#0` inactive-region events), then re-applies the write directly to `ddr_mem0`/`ddr_mem1` with correct byte-lane masking:

```systemverilog
initial begin : ddrc_wr_fix
    int unsigned nb, boff;
    logic [39:0] ba;
    logic [31:0] wa;
    logic [ 7:0] bd;
    logic [31:0] tmp_word;
    forever begin
        @(posedge dut.zynq_ultra_ps_e_0.inst.ddrc.wr_req);
        #1;  // past all #0 inactive-region assignments
        nb = int'(dut.zynq_ultra_ps_e_0.inst.ddrc.wr_bytes);
        for (int b = 0; b < nb; b++) begin
            if (dut.zynq_ultra_ps_e_0.inst.ddrc.wr_strb[b]) begin
                ba   = dut.zynq_ultra_ps_e_0.inst.ddrc.wr_addr + 40'(b);
                wa   = ba[33:2];
                boff = int'(ba[1:0]);
                bd   = dut.zynq_ultra_ps_e_0.inst.ddrc.wr_data[b*8 +: 8];
                if (wa[28] == 1'b0) begin
                    tmp_word = dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem0[wa[27:0]];
                    tmp_word[boff*8 +: 8] = bd;
                    dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem0[wa[27:0]] = tmp_word;
                end else begin
                    tmp_word = dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem1[wa[27:0]];
                    tmp_word[boff*8 +: 8] = bd;
                    dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem1[wa[27:0]] = tmp_word;
                end
            end
        end
    end
end
```

This does not modify any Xilinx IP source and survives IP regeneration. All four test-stand benches carry this block.

**Shadow variant (ConvKernel, PoolingKernel benches).** The block above only re-applies strobed bytes, so it cannot undo stale data the racing write put into *unstrobed* bytes. `conv_tb.sv` and `pooling_tb.sv` keep a byte shadow (`logic [7:0] shadow_mem [longint unsigned]`) filled by every bench `write_mem` chunk (`shadow_store()`) and every strobed beat; for an unstrobed byte that has a shadow entry, the fix block writes the shadow value back instead of skipping it. This makes the benches' `y` tail-pad checks a true test of the kernel's WSTRB.

### Related: late commit of the last write

The DDR model can commit a kernel's final (partial-strobe) write late enough that it lands on top of the *next* test's backdoor-loaded inputs at the same address (MatmulKernel, 2026-09-25: `B[49922]` of test 33 read back as `C[98]` of test 32, never when the case ran alone; a 20 µs settle did not help). `matmul_tb.sv` alternates the DDR base between consecutive tests (`addr_a = 0x1000_0000 + (index % 2) · 0x0400_0000`) so a test's inputs never occupy addresses the previous kernel wrote. See `doc/kernels/MATMUL_OPTIMISATION.md`.

---

## 4. `#0` Delay Semantics in SystemVerilog

`#0` inside an `always @(posedge clk)` block suspends the process to the **inactive event region**. Key properties:

- All **active region** events complete before inactive-region events begin.
- Multiple `#0` suspensions within a single always-block execute in program order, after all other active-region events at the same timestamp have settled.
- An assignment immediately after `#0` is therefore guaranteed to land **after** any concurrent `always @(posedge clk)` that does not itself use `#0`.
- If another always-block also uses `#0` and fires in the same inactive region, relative ordering is tool-dependent (non-deterministic between inactive-region processes).

The arb_wr_6 bug exploits this: DDRC's sampling of `wr_req` is in the active region; `prt_req = 1` (which drives `wr_req`) is also in the active region, but `prt_data` / `prt_strb` are behind `#0`. Moving `prt_req = 1` behind `prt_bytes` (no `#0` after it, but the surrounding context has already consumed several `#0` suspensions) pushes it past the DDRC sampling window.

---

## 5. AXI Beat Alignment Requirement

The PS VIP burst read engine issues full 16-byte (128-bit) aligned beats, and all four kernels now read through 128-bit ports in whole 16-byte words. If the DDR buffer is not padded to a 16-byte boundary, the last partial beat reads uninitialized bytes from adjacent memory, which may be `X` or stale data from a previous test. This manifests as `AXI4_ERRS_RDATA_X` or incorrect output values at the tail of the vector.

**Fix:** Round the region size up to the next 16-byte boundary before calling `fill_const_ddr` / `write_data_ddr` (as every test-stand bench does):

```systemverilog
function automatic int unsigned align_up(int unsigned v, int unsigned a);
    return (v + a - 1) & ~(a - 1);
endfunction

a_bytes = align_up(this.a_count * ELEM_BYTES, 16);
```

The extra bytes (beyond `this.a_count * ELEM_BYTES`) should be filled with a known value (e.g., zero) so the kernel's over-read does not inject garbage into results.

---

## 6. `ap_fixed<16,8>` Encoding Reference

| Meaning | Raw `int16` | Hex |
|---------|------------|-----|
| 1.0     | 256        | `0x0100` |
| 2.0     | 512        | `0x0200` |
| 3.0     | 768        | `0x0300` |
| 6.0     | 1536       | `0x0600` (RELU6 ceiling) |
| −1.0    | −256       | `0xFF00` |
| max     | 32767      | `0x7FFF` |
| min     | −32768     | `0x8000` |

Conversion: `raw = value × 256`, rounded and saturated to int16. The scheduler's encoder (`inference-scheduler/src/dtype.py`) rounds half to even; a C++ assignment to `ap_fixed<16,8>` (default `AP_TRN`) truncates toward −∞.

**Arithmetic reference model (integer, matches HLS AP_TRN truncation).** Historical: an earlier VectorOPKernel bench computed expected values with this function, which predates the `act` register. The current benches compare against C-simulation goldens under `hw/test_data/{vecop,matmul,conv,pool}_test_data/` (VectorOP: `TestSimulation --dump-data`, target `gen_vectorop_test_data`), so `compute_op_result` no longer exists in any bench.

```systemverilog
function automatic logic [15:0] compute_op_result(
    int unsigned op, logic [15:0] a_raw, logic [15:0] b_raw);
    logic signed [31:0] a, b, r;
    a = 32'(signed'(a_raw));
    b = 32'(signed'(b_raw));
    case (op)
        OP_ADD:   r = a + b;
        OP_SUB:   r = a - b;
        OP_MUL:   r = (a * b) >>> 8;     // AP_TRN: truncate toward -inf
        OP_DIV:   r = (b == 0) ? 32'sd0 : (a * 32'sd256) / b;
        OP_RELU:  r = (a >= 0) ? a : 32'sd0;
        OP_RELU6: r = (a < 0) ? 0 : (a > 1536) ? 1536 : a;
        default:  r = 32'sd0;
    endcase
    // saturate to int16 range
    if (r >  32'sd32767) r =  32'sd32767;
    if (r < -32'sd32768) r = -32'sd32768;
    return r[15:0];
endfunction
```

---

## 7. DDRC Module Internals

Source file: `zynq_ultra_ps_e_vip_v1_0_vl_rfs.sv` (all line numbers reference this file).

### 7.1 Module Hierarchy

```
zynq_ultra_ps_e_vip_v1_0_22          (top, ~line 9260)
└── ddrc : zynq_ultra_ps_e_vip_v1_0_22_ddrc        (~line 4233, instantiated ~line 11751)
    ├── ddr_write_ports : zynq_ultra_ps_e_vip_v1_0_22_arb_wr_6   (~line 332,  instantiated ~line 4433)
    ├── ddr_read_ports  : zynq_ultra_ps_e_vip_v1_0_22_arb_rd_6   (symmetric to arb_wr_6)
    └── ddr             : zynq_ultra_ps_e_vip_v1_0_22_sparse_mem (~line 1433, instantiated ~line 4551)
```

Hierarchy path from a testbench DUT named `dut` with a PS instance `zynq_ultra_ps_e_0`:

```
dut.zynq_ultra_ps_e_0.inst.ddrc               ← DDRC module
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_req        ← arbiter output (write-request to DDRC)
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_addr       ← 40-bit byte address
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_data       ← 32768-bit (4096-byte) burst data
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_strb       ← 4096-bit byte strobe
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_bytes      ← 13-bit byte count [12:0]
dut.zynq_ultra_ps_e_0.inst.ddrc.wr_ack        ← DDRC → arbiter acknowledge
dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem0  ← sparse 32-bit array, bank 0
dut.zynq_ultra_ps_e_0.inst.ddrc.ddr.ddr_mem1  ← sparse 32-bit array, bank 1
```

### 7.2 DDRC Module Interface (~lines 4233–4327)

```verilog
module zynq_ultra_ps_e_vip_v1_0_22_ddrc(
    rstn,         // active-low reset
    sw_clk,       // system clock (all DDRC logic clocked on posedge)
    // 6× write ports (port0..port5):
    ddr_wr_ack_portN,    // output: ack to port N
    ddr_wr_dv_portN,     // input:  data-valid (write request) from port N
    ddr_wr_addr_portN,   // input:  40-bit byte address
    ddr_wr_data_portN,   // input:  burst write data
    ddr_wr_strb_portN,   // input:  byte strobes
    ddr_wr_bytes_portN,  // input:  byte count
    ddr_wr_qos_portN,    // input:  QoS
    // 6× read ports (port0..port5):
    ddr_rd_req_portN,    // input:  read request
    ddr_rd_dv_portN,     // output: read data valid
    ddr_rd_addr_portN,   // input:  40-bit byte address
    ddr_rd_data_portN,   // output: burst read data
    ddr_rd_bytes_portN,  // input:  byte count
    ddr_rd_qos_portN     // input:  QoS
);
```

### 7.3 Key Internal Parameters (`zynq_ultra_ps_e_vip_v1_0_22_local_params.sv`, ~lines 21–116)

| Parameter              | Value     | Meaning |
|------------------------|-----------|---------|
| `addr_width`           | 40        | Byte address bits |
| `data_width`           | 32        | Memory word width (bits) |
| `max_burst_len`        | 256       | Maximum AXI burst length |
| `max_data_width`       | 128       | AXI data bus width (bits) |
| `max_burst_bits`       | 32768     | Max burst data buffer: 256 × 128 |
| `max_burst_bytes`      | 4096      | Max burst data in bytes |
| `max_burst_bytes_width`| 12        | log2(4096); byte-count fields are declared `[max_burst_bytes_width:0]` = 13 bits |
| `mem_width`            | 4         | Bytes per memory word |
| `shft_addr_bits`       | 2         | log2(mem_width): word address shift |

### 7.4 Internal Write/Read Bus Signals (~lines 4419–4431)

These wires connect `arb_wr_6` output to the DDRC state machine:

```verilog
wire                       wr_req;                      // write request (from arbiter)
wire [max_burst_bits-1:0]  wr_data;                     // 32768-bit burst payload
wire [max_burst_bytes-1:0] wr_strb;                     // 4096-bit byte-enable mask
wire [addr_width-1:0]      wr_addr;                     // 40-bit byte address
wire [max_burst_bytes_width:0] wr_bytes;                // 13-bit byte count [12:0]
reg                        wr_ack;                      // acknowledge back to arbiter

reg  [max_burst_bits-1:0]  rd_data;                     // read result
wire [addr_width-1:0]      rd_addr;
wire [max_burst_bytes_width:0] rd_bytes;
reg                        rd_dv;                       // read data valid
wire                       rd_req;
```

### 7.5 DDRC State Machine (~lines 4553–4589)

The DDRC core is a 2-state `always @(posedge sw_clk)` FSM:

```verilog
always @(posedge sw_clk or negedge rstn) begin
    if (!rstn) begin
        wr_ack <= 0; rd_dv <= 0; state <= 2'd0;
    end else begin
        case (state)
        0: begin  // IDLE
            wr_ack <= 0; rd_dv <= 0;
            if (wr_req) begin
                ddr.write_mem(wr_data, wr_addr, wr_bytes, wr_strb);
                wr_ack <= 1;
                state  <= 1;
            end
            if (rd_req) begin
                ddr.read_mem(rd_data, rd_addr, rd_bytes);
                rd_dv  <= 1;
                state  <= 1;
            end
        end
        1: begin  // ACK PULSE — deassert for 1 cycle then return to IDLE
            wr_ack <= 0; rd_dv <= 0;
            state  <= 0;
        end
        endcase
    end
end
```

**Important:** The FSM samples `wr_req` on the **same posedge** at which `arb_wr_6` may be asserting it. If `wr_req` arrives in the active event region (the race condition described in §3), the FSM calls `write_mem()` with whatever `wr_addr`/`wr_data`/`wr_strb`/`wr_bytes` are currently on the wires — which may be stale from the previous burst.

The FSM does **not** double-buffer or re-check parameters: one call to `ddr.write_mem()` is made synchronously within the always-block, completing before the next posedge.

### 7.6 `zynq_ultra_ps_e_vip_v1_0_22_sparse_mem` — DDR Memory Model (~lines 1433–1900)

#### Array Declarations (~lines 1451–1452)

```verilog
parameter mem_size      = 32'h4000_0000;  // 1 GB per bank
parameter xsim_mem_size = 32'h1000_0000;  // 256 MB (×4 for xsim/isim)

reg /*sparse*/ [data_width-1:0] ddr_mem0 [0:(mem_size/mem_width)-1];
// Covers byte range 0x0_0000_0000 – 0x0_3FFF_FFFF (268,435,456 × 32-bit words)

reg /*sparse*/ [data_width-1:0] ddr_mem1 [0:(mem_size/mem_width)-1];
// VIP source comment says 0x8_0000_0000 – 0x8_3FFF_FFFF, but the indexing
// below (word_addr[28]) maps byte range 0x4000_0000 – 0x7FFF_FFFF here
```

Both arrays are declared `/*sparse*/` — xsim allocates storage only for written addresses, so unwritten locations return `X` (not 0) when read back.

#### Bank Selection

Address bit [28] of the **word address** (`byte_addr >> 2`) selects the bank:

```verilog
task automatic get_data;
    input  [addr_width-1:0] addr;   // byte address
    output [data_width-1:0] data;
    begin
        if (addr[28] == 1'h0)
            data = ddr_mem0[addr[27:0]];
        else
            data = ddr_mem1[addr[27:0]];
    end
endtask
```

The index passed is `addr[27:0]` where `addr` is already the **word address** (`byte_addr >> 2`). In practice from a 40-bit byte address `ba`:

```systemverilog
wa    = ba[33:2];          // word address (32-bit)
bank  = wa[28];            // 0 → ddr_mem0, 1 → ddr_mem1
index = wa[27:0];          // array index within selected bank
```

#### Public Tasks on `sparse_mem`

| Task | Signature | Purpose |
|------|-----------|---------|
| `write_mem` | `(data, start_addr, no_of_bytes, strb)` | Byte-strobe write, called by DDRC FSM |
| `read_mem`  | `(data, start_addr, no_of_bytes)` | Burst read, called by DDRC FSM |
| `pre_load_mem` | `(data, start_addr, no_of_bytes)` | Pre-load without strobe (full-byte) |
| `pre_load_mem_from_file` | `(filename, start_addr)` | Load from hex file |
| `peek_mem_to_file` | `(filename, start_addr, no_of_bytes)` | Dump memory to file |
| `wait_mem_update` | `(addr)` | Block until address is written |

`write_mem` operates byte-wise internally, applying the `strb` mask bit per byte. The `ddrc_wr_fix` workaround bypasses `write_mem` and directly indexes `ddr_mem0`/`ddr_mem1` to avoid the race; this is safe because the sparse arrays are plain SystemVerilog registers accessible anywhere.

### 7.7 `zynq_ultra_ps_e_vip_v1_0_22_arb_wr_6` — Write Arbiter (~lines 332–838)

Six-port priority-round-robin arbiter. Each port presents:

| Port-N signal | Direction | Meaning |
|---------------|-----------|---------|
| `prt_dvN`     | input     | Data valid / write request |
| `prt_dataN`   | input     | Burst write data (32768 bit) |
| `prt_strbN`   | input     | Byte strobes (4096 bit) |
| `prt_addrN`   | input     | 40-bit byte address |
| `prt_bytesN`  | input     | 13-bit byte count |
| `prt_qosN`    | input     | 4-bit QoS |
| `prt_ackN`    | output    | Acknowledge (pulse) |

Merged output to DDRC: `prt_req`, `prt_data`, `prt_strb`, `prt_addr`, `prt_bytes`, `prt_qos`; acknowledge back: `prt_ack`.

**State machine states:** `wait_req`, `serv_req0`…`serv_req5`, `wait_ack_low`

The race condition (§3) originates in the `wait_req` and `serv_reqN` states where `prt_req = 1` precedes the `#0`-delayed data assignments (~lines 425–486):

```verilog
wait_req: begin
    if (prt_dv0) begin
        state     = serv_req0;
        prt_req   = 1;           // ← RACE: active region, DDRC samples this NOW
        prt_qos   = prt_qos0;
        #0; prt_data  = prt_data0;  // ← settles in inactive region
        #0; prt_strb  = prt_strb0;
        prt_addr  = prt_addr0;
        prt_bytes = prt_bytes0;
    end else if (prt_dv1) begin
        // ... same pattern for ports 1-5
    end
end
```

### 7.8 Clock Generation

The `sw_clk` driving the DDRC is produced by `zynq_ultra_ps_e_vip_v1_0_22_gen_clock` (~line 4173). The PS VIP generates its own internal clock independent of the PL fabric clock; the DDRC FSM, both arbiters, and all AXI port logic run on this clock.

---

## 8. Relevant File Paths

| File | Purpose |
|------|---------|
| `/mnt/data/xilinx/2025.2/data/ip/xilinx/zynq_ultra_ps_e_vip_v1_0/hdl/zynq_ultra_ps_e_vip_v1_0_vl_rfs.sv` | PS VIP source — contains `arb_wr_6` |
| `/mnt/data/xilinx/2025.2/data/ip/xilinx/zynq_ultra_ps_e_vip_v1_0/hdl/zynq_ultra_ps_e_vip_v1_0_22_local_params.sv` | VIP parameters (§7.3) |
| `/home/ivan/vivado_projects/fix_arb_wr6_prt_req.py` | Python patch script for Fix Option A (outside the repo; not applied) |
| `hw/cormorant_test_stand/kernels/vector_op_test/vector_op_test.srcs/sim_1/new/vectorop_tb.sv` | VectorOPKernel bench — `ddrc_wr_fix`, chunked `fill_const_ddr` / `write_data_ddr`, per-test scoreboard |
| `hw/cormorant_test_stand/kernels/matmul_op_test/matmul_op_test.srcs/sim_1/new/matmul_tb.sv` | MatmulKernel bench — `ddrc_wr_fix`, alternating DDR base (§3) |
| `hw/cormorant_test_stand/kernels/conv_test/conv_test.srcs/sim_1/new/conv_tb.sv` | ConvKernel bench — `ddrc_wr_fix` with shadow restore of unstrobed bytes, `y` tail-pad check |
| `hw/cormorant_test_stand/kernels/pooling_test/pooling_test.srcs/sim_1/new/pooling_tb.sv` | PoolingKernel bench — `ddrc_wr_fix` with shadow restore, `y` tail-pad check |
| `hw/test_data/{vecop,matmul,conv,pool}_test_data/` | Golden fixtures the benches read (`make behavior_test_<k>`) |
