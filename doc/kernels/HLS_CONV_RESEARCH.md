# HLS Convolution Kernels: State of the Art for KV260-class Devices

> Background literature survey (2026-05-13), not a plan.  The target context
> (300 MHz) is aspirational: HLS targets 150 MHz and the board runs at 250 MHz
> (since FMAX_250_PLAN; 100 MHz when this was written).
> What the kernel implements: CONV_KERNEL.md; history: CONV_OPTIMISATION.md.
> §2's PM×PN grid became the 16×16 grid (§2.24 / §2.40) and PX = 2 (§2.42);
> int8 packing (§6), Winograd (§8) and DW→PW fusion (§5) are unexplored.

Target context: Vitis HLS 2025.2, XCK26 Zynq UltraScale+ MPSoC, 300 MHz, `ap_fixed<16,8>`. The KV260 has ~1248 DSP48E2 slices and ~144 BRAM tiles (≈ 5 MB on-chip) — small enough that weight buffering is rarely "free" and big enough that streaming/tiled designs win. This document focuses on techniques that change the QoR meaningfully, not introductory material.

## 1. Loop Ordering, Tiling and the Roofline Frame

The canonical analytical frame is still Zhang/Cong, FPGA'15, _Optimizing FPGA-based Accelerator Design for Deep CNNs_ ([paper](https://dl.acm.org/doi/abs/10.1145/2684746.2689060)). The seven-deep convolution loop nest

```
for to (M): for ti (N): for r (R): for c (C): for i (K): for j (K):
  out[to][r][c] += in[ti][r*S+i][c*S+j] * w[to][ti][i][j]
```

is tiled into block sizes `Tm` (output channels), `Tn` (input channels), `Tr` (output rows), `Tc` (output cols). Computation roof is `2 * Tm * Tn / II` MACs/cycle. Memory traffic per tile scales as `αin·Tn·(Tr+K-1)·(Tc+K-1) + αw·Tm·Tn·K² + αout·Tm·Tr·Tc`. The roofline picks the `(Tm,Tn,Tr,Tc)` quadruple that puts the design at the compute–bandwidth knee — past it more DSPs starve, before it BRAMs are wasted. For each layer the optimum differs; either the architecture is built for the worst layer or pipelined per-layer (cf. fpgaConvNet).

Practical loop ordering on KV260 with ap_fixed<16,8>:

- **Output-stationary `(to, ti, r, c, i, j)` with `ti` innermost of the inner band.** Accumulator lives in a register; partial sums never spill to BRAM; long input-channel chain amortises the input read.
- Keep `i,j` (kernel) as the inner two loops only if `K` is small and fixed — otherwise unroll them entirely so the schedule sees `K²` independent multiplies. The Vivado HLS convolution tutorial uses exactly this shape ([tutorial](https://xilinx.github.io/Vitis-Tutorials/2021-1/build/html/docs/Hardware_Acceleration/Design_Tutorials/01-convolution-tutorial/lab2_conv_filter_kernel_design.html)).
- The accumulator dependence on `ti` triggers a false II=2 warning if HLS cannot prove independence; address with `#pragma HLS DEPENDENCE variable=acc inter false` or by accumulating into a register (scalar) inside the unrolled fan-in tree. Discussion in UG1399 _Managing Pipeline Dependencies_ ([docs](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Managing-Pipeline-Dependencies)).

Dataflow choices (Eyeriss-style taxonomy, also used in Caffeine and Angel-Eye):

| Dataflow | Stationary in PE | Best when | Cost |
|---|---|---|---|
| Output-stationary | partial sum | DSP-limited, many input channels | reload weights/inputs every spatial step |
| Weight-stationary | filter weights | weights fit in PE registers, large activations | reload psum |
| Input-stationary | activation window | row reuse, large `K` | reload weights and psums |

For ap_fixed<16,8> on an XCK26 with 1248 DSPs and tight BRAM, **output-stationary + tiled along (Tm,Tn)** is the dominant choice — it is what Caffeine ([IEEE TCAD](https://ieeexplore.ieee.org/document/8497058/)) and Angel-Eye ([TCAD 2017](https://ieeexplore.ieee.org/document/7930521/)) ship.

## 2. Parallelization Axes — PM, PN, PX/PY

Three parallel axes carry the DSP count:

- **PM (output-channel parallelism)** — unroll `to` by PM. Each PE has its own weight, shares input activations. Excellent activation reuse, multiplies BRAM ports needed for weights by PM. Typical PM = 8…32.
- **PN (input-channel parallelism)** — unroll `ti` by PN inside an adder tree. Each PE multiplies a different input channel by a different weight; results sum each cycle. PN is what gives output-stationary its compute density. Typical PN = 8…16.
- **PX/PY (spatial parallelism)** — unroll over `r` or `c`. Produces multiple output pixels per cycle from a shared weight, replicating the window buffer reads. Cheap on weight ports, expensive on activation ports.

DSP count ≈ `PM · PN · PX · PY`. For a 16x16 PE grid (PM=PN=16) at 300 MHz you target 153.6 GMAC/s — close to the KV260 sweet spot of ~600 GOP/s (1 MAC = 2 OPs). Caffeine reports unrolling PM=64, PN=7 for AlexNet on a Zynq-7045 ([Caffeine](https://ieeexplore.ieee.org/document/8497058/)).

The HLS pattern that synthesizes cleanly:

```cpp
for (int to_o = 0; to_o < M; to_o += PM) {
  for (int ti_o = 0; ti_o < N; ti_o += PN) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
        acc_t acc[PM];
#pragma HLS ARRAY_PARTITION variable=acc complete
        for (int i = 0; i < K; ++i)
          for (int j = 0; j < K; ++j)
            for (int tm = 0; tm < PM; ++tm) {
#pragma HLS UNROLL
              for (int tn = 0; tn < PN; ++tn) {
#pragma HLS UNROLL
                acc[tm] += in_buf[ti_o+tn][r+i][c+j]
                         * w_buf [to_o+tm][ti_o+tn][i][j];
              }
            }
        // write acc[0..PM-1] to out_buf
      }
    }
  }
}
```

Two non-obvious rules:

1. **Partition the data dimensions you unroll along.** `ARRAY_PARTITION dim=1 cyclic factor=PN` on `in_buf`, `dim=1 cyclic PM` and `dim=2 cyclic PN` on `w_buf`. Without partitioning, BRAM port count caps PE count at 2 per array and you get _"failed to schedule"_ at II=1 ([AMD partition docs](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition)).
2. **Cyclic vs block:** if the unroll index is the fastest-moving (typical) use **cyclic** so consecutive lanes hit different banks; block-partition only when the unroll index is the slow tile counter.

## 3. Line / Window Buffers

For II=1 sliding-window convolution the recipe is:

```cpp
data_t line_buf[K-1][W];                          // K-1 full rows
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
data_t window  [K][K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0   // full
```

Each cycle: shift `window` left by one column, push the new input pixel through `line_buf` (the oldest column of `line_buf` enters the bottom-left of `window`), pop one output. The HLS Tiny Tutorials 2D convolution example and the `hls::LineBuffer`/`hls::Window` templates in `hls_video.h` / Vitis Vision are the reference ([PYNQ discussion](https://discuss.pynq.io/t/2d-convolution-with-line-buffer-from-hls-tiny-tutorials/2813), [tutorial](https://xilinx.github.io/Vitis-Tutorials/2021-1/build/html/docs/Hardware_Acceleration/Design_Tutorials/01-convolution-tutorial/lab2_conv_filter_kernel_design.html)).

For multi-channel CNN conv, replicate per input channel that you process in parallel — line buffer becomes `line_buf[PN][K-1][W]`, with `dim=1` partitioned `complete` (channel parallel reads) and `dim=2` partitioned `complete` (kernel-row parallel reads). Total cost: `PN · (K-1) · W` data_t cells across `PN·(K-1)` BRAM-18Ks.

Padding: pre-clear `line_buf` and emit `K/2` extra clock cycles before reading real data, gating the output write by a coordinate counter. Stride > 1: emit outputs only on the strided indices (`if (r % S == 0 && c % S == 0) write_output`); HLS keeps the pipeline running anyway. _Avoid_ trying to skip cycles — that breaks II=1.

## 4. Memory Hierarchy and AXI

KV260 PS↔PL DDR bandwidth caps near ~10 GB/s through the HP ports. ap_fixed<16,8> at 300 MHz on a 512-bit AXI bus = 19.2 GB/s — bus, not DRAM, is your headroom. Burst widening is therefore the first real optimization.

- **`#pragma HLS INTERFACE m_axi port=in bundle=g0 offset=slave max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=8`** — explicit is safer than the auto-widener. Defaults are 16-beat burst, 2 outstanding; both leave bandwidth on the table ([Ramon Heras' Vitis HLS AXI burst notes](https://ramonheras.com/posts/axim-2/), [pp4fpgas AXI lab](https://pp4fpgas.readthedocs.io/en/latest/axi4.html)).
- **Use `memcpy` (or a deterministic `for` with constant-bound trip count) to feed a local on-chip array** — HLS infers a clean burst. A variable-bound `for` with `if (...) break` defeats burst inference. Vitis-Tutorials Bloom kernel illustrates the canonical 512-bit dataflow pattern ([Bloom tutorial](https://xilinx.github.io/Vitis-Tutorials/2022-1/build/html/docs/Hardware_Acceleration/Design_Tutorials/02-bloom/4_implement-kernel.html)).
- **Adapter pattern:** `read512 → resize → hls::stream<vec<PN>> → compute → resize → write512`. Width adaptation lives in dedicated `void mem_read(...)` / `void mem_write(...)` functions so DATAFLOW sees stable producer-consumer relations.

Double buffering inside DATAFLOW:

```cpp
void conv_top(...) {
#pragma HLS DATAFLOW
  hls::stream<vec_t> in_s, w_s, out_s;
#pragma HLS STREAM variable=in_s depth=2*Tn*(Tr+K-1)
  load_input (in_ddr,  in_s);
  load_weight(w_ddr,   w_s);
  compute    (in_s, w_s, out_s);
  store_output(out_s, out_ddr);
}
```

DATAFLOW gives implicit ping-pong via the stream FIFOs. A known sharp edge: a DATAFLOW process that writes to `m_axi` with non-trivial bursts can lose outstanding write parallelism ([issue #6](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/issues/6)) — when this bites, split the store into a separate function that does its own `memcpy`. For arrays internal to DATAFLOW use `#pragma HLS BIND_STORAGE type=ram_t2p impl=bram` or `#pragma HLS ARRAY_PARTITION ... type=cyclic ...` to control banking.

**On-chip weight buffering:** XCK26's ~5 MB BRAM holds maybe two ResNet-18 layers' worth of weights at int8 — MobileNet-v1 weights (~4.2 M) fit entirely if quantised. When a layer's weights fit, hoist the `load_weight` out of the spatial loop and keep weights stationary across the whole feature map — eliminates DDR weight traffic almost completely and was the main optimisation in Angel-Eye for VGG layers ([Angel-Eye TCAD](https://ieeexplore.ieee.org/document/7930521/)).

## 5. Depthwise & Depthwise-Separable

Depthwise (DW) has arithmetic intensity `K²` MACs per input byte (≈ 9 for 3×3) vs ~`K²·M` for standard conv. A 1024-DSP grid built for standard conv runs DW at < 5% utilisation. The literature converges on three fixes:

1. **Channel-parallel DW** — instead of unrolling along input channel (there is no `ti` reduction in DW), unroll along the depthwise channel index. Each lane is an independent `K×K` MAC tree, sharing only the line-buffer plumbing. Bai et al., FPGA'18 _"A CNN Accelerator on FPGA Using Depthwise Separable Convolution"_ uses 32-wide channel parallel slices, each a 3×3 multiplier array sharing a 32-channel line buffer; reported 17.6 GOPS and 75.9% DSP utilisation on Arria-10 ([arxiv 1809.01536](https://ar5iv.labs.arxiv.org/html/1809.01536)).
2. **Fused DW→PW** — write DW outputs directly into the pointwise (PW) engine's input FIFO. Activation never touches DDR. Knapheide & Stabernack ([IEEE FPL 2020](https://ieeexplore.ieee.org/document/9221517/)) report 266 fps MobileNet-v2 with this fusion. Compatible with DATAFLOW: DW kernel and PW kernel are two functions linked by an `hls::stream<vec<C>>`.
3. **Reuse the standard-conv engine for PW** — PW (1×1) is just GEMM along channels; the existing standard-conv PE grid runs PW at full utilisation. Only the DW stage needs its own (smaller, channel-parallel) engine. Mobile-X ([IEEE Access 2024](https://ieeexplore.ieee.org/document/10630707/)) and the MDPI 2023 design ([MDPI](https://www.mdpi.com/2079-9292/12/7/1571)) both adopt this split-engine model.

Engine-idling antidotes:
- **Co-schedule DW with the previous PW's tail** — while PW is finishing tile `i`, DW starts on tile `i-1`. Two functions in DATAFLOW.
- **Wider DW channel parallelism than PW** — DW is bandwidth-bound; throw extra cheap lanes at it (no MAC tree depth, just K² MACs). PW gets the deep adder tree.
- **Squeeze-and-excite & batchnorm folding** — fold BN into the PW weights offline; SE multipliers are tiny relative to convs ([MDPI AI 2025](https://www.mdpi.com/2673-2688/6/10/244)).

For your `ap_fixed<16,8>` baseline, the cheapest DW pattern is: `PN_dw = 16` lanes, each a 3×3 fully-unrolled tree (9 DSPs / lane), so DW consumes 144 DSPs and runs at one output pixel × 16 channels per cycle.

## 6. Quantization-Aware Design

DSP48E2 on UltraScale+ has a 27×18 multiplier and a 48-bit accumulator. Two int8 MACs per DSP via the _shared-weight packing trick_ (XLNX WP486 [_Deep Learning with INT8 Optimization on Xilinx Devices_](https://www.xilinx.com/support/documentation/white_papers/wp486-deep-learning-int8.pdf), [Edge-AI summary](https://www.edge-ai-vision.com/2016/11/deep-learning-with-int8-optimization-on-xilinx-devices/)):

```
pre = (a << 19) | b           // a and b are 8-bit, separated in 27-bit pre-adder result
prod = pre * w                // single DSP multiply, w is 8-bit
psum_a = prod[47:19]          // high half
psum_b = prod[18:0]           // low half
```

Cost: ~1 extra DSP every 7 to prevent oversaturation → effective 1.75× MAC density (Xilinx's reported figure). HLS gets this through `ap_int<8>` operand types plus DSP-packing templates — straight `ap_fixed<16,8>` × `ap_fixed<16,8>` will _not_ pack. If you can drop to int8 weights+activations for the standard conv layers and keep ap_fixed<16,8> only for sensitive ones, you get a near-2× DSP scale-up for free.

Survey of further DSP packing schemes including INT4 and mixed precision is in DSP-Packing ([arxiv 2203.11028](https://arxiv.org/pdf/2203.11028)). Vitis AI's DPUCZDX8G (the KV260 DPU) is built around exactly this INT8 + DSP-packing scheme ([Vitis AI 3.0 system integration](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html)).

## 7. Published Open-Source References

| Project | Approach | Useful pattern |
|---|---|---|
| [Vitis-HLS-Introductory-Examples](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples) | Burst inference, port widening, dataflow | Concrete pragma recipes; the `Interface/Memory/manual_burst/*` cases are the canonical burst patterns |
| Vitis Vision (xfopencv) | `hls::LineBuffer`, `hls::Window` | Production-quality sliding-window primitives — read the headers, the loop shapes generalise |
| [hls4ml](https://github.com/fastmachinelearning/hls4ml) | Per-layer parallelism, fully unrolled when it fits | `Conv2D` resource strategy uses `pragma HLS ARRAY_PARTITION dim=0 complete` plus per-output adder trees. ACM TRETS 2025 reference ([paper](https://dl.acm.org/doi/10.1145/3801979)) |
| [FINN](https://github.com/Xilinx/finn) (BNN-PYNQ, finn-hlslib) | Streaming, MVAU/VVAU compute units, low-precision packing | `MatrixVectorActivationUnit` template; relevant if you ever do BNN / 4-bit |
| [fpgaConvNet](https://fpgaconvnet.com/about.html) | SDF graph, per-layer pipelined | Architecture exploration approach — uses HLS underneath, the SDF transformation framework is the published part |
| Vitis-Tutorials 2D conv | Dataflow + line buffer template | Reference for the standalone 3×3 sliding-window kernel |
| [WinoGen DAC'24](https://dl.acm.org/doi/10.1145/3649329.3657392) | Winograd F(m,3) HLS IP generator | Source of Winograd matrix code that survives synthesis |
| Bai et al., FPGA'18 ([arxiv](https://ar5iv.labs.arxiv.org/html/1809.01536)) | Split DW/PW, MME unit | Concrete DW micro-architecture (32-channel slice, 3×3 unrolled) |
| Caffeine ([TCAD 2018](https://ieeexplore.ieee.org/document/8497058/)) | Uniform GEMM representation for conv + FC | Bandwidth optimisation chapter is the reference for memory-access reorganisation |
| [DPU-V4E arxiv 2506.11441](https://arxiv.org/pdf/2506.11441) | Vitis-AI DPU internals | What an industrial PE array looks like, instruction-level |

## 8. Winograd: when it pays

F(2,3) needs 4 muls instead of 6 for a 1D 3-point conv (1.5×); F(2×2, 3×3) needs 16 muls for a 2×2 output tile instead of 36 (2.25×); F(4×4, 3×3) drops to 36 muls per 4×4 output, a 4× theoretical reduction. Practical wins on FPGAs:

- 3×3 layers only (5×5 has a much worse Winograd ratio; 1×1 has none).
- Transform matrices `G`, `B`, `A` are constant — HLS unrolls them away if all dims are compile-time. Don't even try Winograd with runtime-variable layer geometry.
- Numerical inflation: F(4×4, 3×3) requires ≥ 2 extra bits of headroom on accumulators; ap_fixed<16,8> activations may need an extra 1–2 bits of integer to keep saturation away. Lu/Liang FCCM'17 and SpWA FPGA'18 ([SpWA](https://ceca.pku.edu.cn/docs/20181119154841593887.pdf)) report 2-3× layer-level speedups on real ZCU-class boards.

If your conv kernel mostly serves MobileNet/EfficientNet, Winograd is _not_ the right battle — those have very few 3×3 standard convs and many 1×1 + DW. If you ship VGG/ResNet/YOLO, Winograd on the 3×3 layers is worth the complexity.

## 9. Common HLS Pitfalls

1. **Burst not inferred.** Cause: variable-bound loop, conditional break, or interleaved access pattern. Fix: extract the bulk transfer into its own function with a constant-bound `memcpy` or simple `for`; check the synthesis log for "_Burst inferred_" / "_Burst failed_" on every m_axi port ([Heras notes](https://ramonheras.com/posts/axim-2/)).
2. **II=2 from a false accumulator dependence.** HLS sees `acc[idx] += ...` and assumes `idx` collides across iterations. Fix: use `#pragma HLS DEPENDENCE variable=acc inter false` OR keep the accumulator as a scalar inside the inner pipelined loop and write to the array only at the band boundary ([UG1399](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Managing-Pipeline-Dependencies)).
3. **"Failed to schedule" with a fully-unrolled inner loop.** Usually a BRAM port shortage. Add `ARRAY_PARTITION` matching the unroll factor on every array indexed by the unrolled iterator. For PM × PN, partition output dim by PM and weight dims by PM and PN.
4. **Long combinational chains from a flat reduction.** HLS infers a balanced adder tree _if_ associativity is provable (true for integers / ap_fixed). Float requires `#pragma HLS BIND_OP op=fadd ... latency=...` or explicit tree rewriting; otherwise the chain serialises.
5. **DATAFLOW + m_axi store with no outstanding writes.** Symptom: throughput halves on writes. Workaround: dedicated store function, `num_write_outstanding=8`, `max_write_burst_length=256` ([issue 6](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/issues/6)).
6. **Cyclic vs block partition picked wrong.** If you partition cyclic with factor F but iterate with stride F, every access lands in bank 0. Match the partition pattern to the access pattern (cyclic for unit-stride parallel access, block for tile-strided access).
7. **`#pragma HLS PIPELINE` on the wrong loop.** Putting it on the outermost loop forces HLS to flatten and unroll everything inside — explodes resources. Put it on the loop whose II target equals one MAC per cycle, leave outer loops as sequential.
8. **Stream depth too small under DATAFLOW.** HLS infers depth 2 by default; a producer that bursts and a consumer that slowly drains deadlock. Size stream depths to the largest tile they buffer (`#pragma HLS STREAM variable=s depth=...`).
9. **AXI offset address not 4 KB-aligned.** Burst splits at 4 KB boundaries per AMBA spec; misaligned base means every burst is shorter than expected. Allocate buffers `posix_memalign(..., 4096, ...)` on the PS side.
10. **Forgetting `set_directive_top` matches the C name.** Tcl synthesis silently builds a different (smaller) IP.

## 10. Putting it together for KV260 + ap_fixed<16,8>

A reasonable target architecture for the conv kernel on this branch:

- **Standard conv PE grid:** PM=16, PN=16 → 256 DSPs, ~76 GMAC/s at 300 MHz. Output-stationary, with the inner `i,j` fully unrolled when `K ≤ 5`. Use `#pragma HLS DEPENDENCE` to declare the accumulator independence; partition input and weight buffers cyclically along the channel dim by factor 16.
- **Line buffer:** `line_buf[PN][K-1][max_W]` partitioned `complete` on dim 1, `complete` on dim 2; window `K×K` partitioned `complete`.
- **AXI:** three `m_axi` bundles (input, weight, output), `max_widen_bitwidth=512`, `max_read_burst_length=256`, `num_read_outstanding=8`. Separate load_in / load_w / compute / store_out functions glued by `DATAFLOW`, each adapter doing width-conversion.
- **Weight buffer:** sized to hold one `(Tm, Tn, K, K)` tile, ping-pong via DATAFLOW. When a layer's full weights are < 1 MB, load once and run all tiles against the cached copy.
- **Depthwise engine:** separate function instantiated alongside the standard PE grid, 16-channel parallel, each with a fully unrolled 3×3 tree. Output stream fed directly into the standard PE grid's input when fused DW→PW pipelines apply.
- **Hooks for int8 packing later:** parametrise the PE on operand type so a future switch from `ap_fixed<16,8>` to `ap_int<8>` operands enables the DSP-packed 2-MAC/cycle/DSP mode without restructuring the loop nest.

These are the techniques that consistently move QoR on 2023–2025 ZCU/KV-class designs. The big remaining wins are usually quantisation (int8 packing → ~1.75× DSP) and per-layer architecture specialisation (Winograd on 3×3 standard, dedicated DW engine on mobile nets).

## Sources

- Zhang et al., _Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks_, FPGA 2015 — https://dl.acm.org/doi/abs/10.1145/2684746.2689060
- Qiu / Guo et al., _Angel-Eye_, IEEE TCAD 2017 — https://ieeexplore.ieee.org/document/7930521/
- Zhang et al., _Caffeine_, IEEE TCAD 2018 — https://ieeexplore.ieee.org/document/8497058/
- Bai, Zhao, He, _A CNN Accelerator on FPGA Using Depthwise Separable Convolution_, FPGA 2018 — https://ar5iv.labs.arxiv.org/html/1809.01536
- Knapheide & Stabernack, _High Throughput MobileNetV2 FPGA Implementation_, IEEE FPL 2020 — https://ieeexplore.ieee.org/document/9221517/
- _High-Performance FPGA-Based Depthwise Separable Convolution Accelerator_, MDPI Electronics 2023 — https://www.mdpi.com/2079-9292/12/7/1571
- Mobile-X, _Dedicated FPGA Implementation of the MobileNet Accelerator_, IEEE Access 2024 — https://ieeexplore.ieee.org/document/10630707/
- _An Efficient FPGA-based DSCNN Accelerator with Hardware Pruning_, ACM TRETS 2023 — https://dl.acm.org/doi/10.1145/3615661
- _Efficient CNN Accelerator with Optimized DSC and SE Modules_, MDPI AI 2025 — https://www.mdpi.com/2673-2688/6/10/244
- _FPGA-based Acceleration for CNNs: A Comprehensive Review_, arXiv 2025 — https://arxiv.org/html/2505.13461v1
- Xilinx WP486, _Deep Learning with INT8 Optimization on Xilinx Devices_ — https://www.xilinx.com/support/documentation/white_papers/wp486-deep-learning-int8.pdf
- _DSP-Packing_, arXiv 2022 — https://arxiv.org/pdf/2203.11028
- Lu et al., _Evaluating Fast Algorithms (Winograd) for CNNs on FPGAs_, FCCM 2017 — https://ceca.pku.edu.cn/media/lw/6940b5b0e09259131ff19334f3efeecd.pdf
- _SpWA: Sparse Winograd Convolutional NN_, FPGA 2018 — https://ceca.pku.edu.cn/docs/20181119154841593887.pdf
- _WinoGen_, DAC 2024 — https://dl.acm.org/doi/10.1145/3649329.3657392
- hls4ml, ACM TRETS 2025 — https://dl.acm.org/doi/10.1145/3801979
- FINN — https://github.com/Xilinx/finn
- fpgaConvNet — https://fpgaconvnet.com/about.html
- Vitis HLS UG1399 _Managing Pipeline Dependencies_ — https://docs.amd.com/r/en-US/ug1399-vitis-hls/Managing-Pipeline-Dependencies
- Vitis HLS UG1399 _pragma HLS array_partition_ — https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition
- Vitis-Tutorials 2D convolution — https://xilinx.github.io/Vitis-Tutorials/2021-1/build/html/docs/Hardware_Acceleration/Design_Tutorials/01-convolution-tutorial/lab2_conv_filter_kernel_design.html
- Vitis-Tutorials Bloom kernel (DATAFLOW + 512-bit) — https://xilinx.github.io/Vitis-Tutorials/2022-1/build/html/docs/Hardware_Acceleration/Design_Tutorials/02-bloom/4_implement-kernel.html
- Vitis-HLS-Introductory-Examples — https://github.com/Xilinx/Vitis-HLS-Introductory-Examples
- Ramon Heras, _Enabling Burst Transactions on an AXIM Interface in HLS_ — https://ramonheras.com/posts/axim-2/
- pp4fpgas AXI4 lab — https://pp4fpgas.readthedocs.io/en/latest/axi4.html
- Vitis AI 3.0 / DPUCZDX8G — https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html
- DPU-V4E, arXiv 2025 — https://arxiv.org/pdf/2506.11441
