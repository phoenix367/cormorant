# Model Preparation

Stock ONNX models exported from PyTorch / TensorFlow / the ONNX Model Zoo
rarely run through the inference scheduler unchanged. They typically ship
with a dynamic batch dimension (`N`), `BatchNormalization` ops,
dynamic-shape scaffolding (`Shape` → `Gather` / `Concat` → `Reshape`
chains), and tail operators the scheduler doesn't implement (`ArgMax`,
`LogSoftmax`, `TopK`). The scheduler exits with a `SchedulerError` on the
first unsupported op, so most models need a preparation pass before they can
be fed to `inference_scheduler.py`.  (`Constant`, `Identity`, `Dropout`,
`Squeeze` / `Unsqueeze`, `Cast`, `Split` and last-axis `Softmax` are
accepted as they are.)

This doc covers the standard preparation flow built around
`simplify_onnx.py`. For the full list of operators the scheduler accepts,
see [`USER_GUIDE.md`](USER_GUIDE.md#2-supported-onnx-operators).

---

## 1. The one-shot fix: `simplify_onnx.py`

`simplify_onnx.py` is the top-level utility for normalising an ONNX file.
It is enough for most ONNX Model Zoo / torchvision exports:

```bash
python simplify_onnx.py model.onnx --batch 1
```

The default output is `<stem>-simplified.onnx` next to the input.
Pipeline:

1. **Pin input shapes.** `--batch N` rewrites every input's dynamic first
   dim to `N`; `--input-shape NAME=D1,D2,…` (repeatable) handles inputs
   with non-batch dynamic dims, or any time you need a non-1 batch.
2. **`onnxsim.simplify`** — constant folding, dead-node removal, shape
   inference. As a side-effect it usually folds `BatchNormalization` (and
   sometimes also `Squeeze`/`Unsqueeze`, `Reshape`/`Flatten` chains, and
   `Cast` of constants) into surrounding ops.
3. **`onnxoptimizer.fuse_bn_into_conv`** — safety-net pass for any BN
   that onnxsim left in place. Disable with `--no-fuse-bn` if you want
   to inspect the un-fused output.
4. **`onnx.checker.check_model`** — fails loudly on any structural
   issue introduced by the rewrites.
5. **Optional smoke test** — `--check` runs the saved model through
   onnxruntime with a random-input feed and prints output min/max so
   you can sanity-check non-trivial transforms.

The script prints a node-count delta plus a per-op-type diff (changed
counts highlighted in yellow). Examples observed on the models kept in
`inference-scheduler/` (downloaded locally; model files are not tracked in git):

| Source | Nodes (before → after) | BN folded |
|---|---|---|
| `resnet50-v1-12.onnx` | 175 → 122 | 53 → 0 |
| `mobilenet_v1_1.0_224.onnx` | 78 → 59 | 13 → 0 |
| `mobilenetv2-12.onnx` | 105 → 100 | 0 (already folded by exporter) |
| `lenet.onnx` | 18 → 9 | 0 |

Verified by running `simplify_onnx.py <model>.onnx --batch 1` against
each source.  ResNet-18 (`resnet18-v1-7.onnx` / `resnet18-v2-7.onnx`)
is a notable exception: the script leaves it at 69 → 69 with all
BatchNormalization nodes intact, because the BN scale/bias don't
collapse cleanly into the preceding Convs in this export.  A working
49-node ResNet-18 (`resnet18-simplified-fused.onnx`) exists locally
but was produced by a different fusion pipeline; reproducing it
via `simplify_onnx.py` alone is **not currently supported**.

---

## 2. Picking input shapes

`--batch N` is the shortcut: every input whose first dim is dynamic
(`dim_param` set, e.g. `'N'`) gets its first dim set to `N`. Other dims
are left as authored. If any non-first dim is also dynamic, the script
errors out — you have to use `--input-shape`:

```bash
# Single-input vision model with dynamic batch
python simplify_onnx.py resnet50-v1-12.onnx --batch 1
#   data: [N, 3, 224, 224]  →  [1, 3, 224, 224]

# Single-input model with non-batch dynamic dims
python simplify_onnx.py super-resolution-10.onnx \
    --input-shape input=1,1,224,224
```

### Multi-input models

`--input-shape NAME=D1,D2,...` is **repeatable** — pass it once per input
that needs explicit sizing. Order doesn't matter. Inputs not named on
the command line fall back to `--batch` (if given) or stay as authored.

```bash
# Pin every input's shape explicitly
python simplify_onnx.py multi_input_model.onnx \
    --input-shape input_a=1,3,224,224 \
    --input-shape input_b=1,16

# Same model, fully equivalent — order is independent
python simplify_onnx.py multi_input_model.onnx \
    --input-shape input_b=1,16 \
    --input-shape input_a=1,3,224,224
```

`--input-shape` always wins over `--batch` for the named input, so the
two flags compose: `--batch` covers the "vanilla" dynamic-batch inputs
and `--input-shape` overrides the awkward ones.

```bash
# batch=1 everywhere except aux_input, which needs a fixed 64-elem vector
python simplify_onnx.py model.onnx \
    --batch 1 \
    --input-shape aux_input=1,64
```

To discover the input names and current shapes before writing the
command line, dump them with a few lines of `onnx`:

```bash
python - <<'PY'
import onnx
m = onnx.load("model.onnx", load_external_data=False)
for vi in m.graph.input:
    dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or '?')
            for d in vi.type.tensor_type.shape.dim]
    print(f"  {vi.name}: {dims}")
PY
```

Dynamic dims appear as their `dim_param` string (typically `'N'`,
`'batch_size'`, `'sequence'`, …) or `'?'` when no symbol was set.
Anything non-numeric in that listing has to be pinned via `--batch` or
`--input-shape` before the scheduler can consume the model.

---

## 3. Handling unsupported ops left after simplify

`simplify_onnx.py` only does mechanical rewrites — it cannot invent
support for an operator the scheduler doesn't implement. If the scheduler
still reports an unsupported op on a simplified model, the typical
remedies are:

| Symptom | Cause | Fix |
|---|---|---|
| `LogSoftmax`, `ArgMax`, `TopK` in the tail | Classifier head | Strip the tail with `onnx.utils.extract_model(in, out, [model_input], [pre_softmax_tensor])`. A `Softmax` over the last axis runs as a host-CPU op and can stay; the local `mobilenet_v1_1.0_224_no_softmax.onnx` was cut this way before that existed. |
| `BatchNormalization` left after simplify | BN that onnxsim / `fuse_bn_into_conv` could not fold (e.g. `resnet18-v1-7` / `-v2-7`, §1) | Fold BN into the preceding Conv with a custom pass; the scheduler has no BN op. |
| `Shape`, `Gather` / `Concat` on shapes | Dynamic-shape scaffolding | A `simplify_onnx.py` pass with pinned input shapes (`--batch` / `--input-shape`) constant-folds them. |
| `Pad` with non-constant pads | Dynamic padding via `Shape`/`Slice` | Re-export the model with constant padding values, or rewrite via `onnx.compose` / `onnx-graphsurgeon`. |
| `Conv` with `group != 1 and group != in_channels` | Grouped convolution (not depthwise) | Not supported by ConvKernel. The model needs surgery to expand the grouped conv into multiple normal convs. |
| `MatMul` with `k > kMaxK` | Inner-dim larger than the platform's `kernels.matmul.max_k` | Either bump `max_k` in `platforms/<name>.json` (and the RTL kernel's `K_MAX` in `kernels/matmul_rtl/rtl/mm_pkg.sv`) and rebuild the bitstream, or split the matmul along K (manual). |

`onnx.utils.extract_model` is the easiest way to keep just the
"interesting" portion of a network — see the local `mobilenet_v1_*`
sub-graph fixtures (`mobilenet_v1_input_to_avgpool.onnx`,
`mobilenet_v1_conv13_relu6_avgpool.onnx`, etc.) for examples of
intermediate-tensor extraction used to isolate scheduler / kernel bugs
without running the whole network.

---

## 4. Diagnosis workflow

When a model the scheduler refuses to load:

```bash
# 1. Pin shapes + simplify. Most exporter cruft disappears here.
python simplify_onnx.py model.onnx --batch 1 --check

# 2. Try the scheduler on the simplified file.
python inference_scheduler.py model-simplified.onnx --out-dir /tmp/out

# 3. If it fails, the SchedulerError names the offending op and tensor.
#    Inspect the simplified model around that node — `netron`, `onnx.helper`,
#    or `onnx-graphsurgeon` are all useful.
python - <<EOF
import onnx
m = onnx.load("model-simplified.onnx")
for i, n in enumerate(m.graph.node):
    if n.op_type in {"BatchNormalization", "ArgMax", "LogSoftmax", "Shape"}:
        print(i, n.op_type, n.name, "->", list(n.output))
EOF

# 4. Strip the offending region. For a clean tail-cut:
python - <<EOF
import onnx, onnx.utils
onnx.utils.extract_model(
    "model-simplified.onnx", "model-trimmed.onnx",
    input_names=["data"], output_names=["last_supported_tensor"])
EOF

# 5. Re-simplify the trimmed model (so the constant-folding pass sees the
#    new, shorter graph) and re-run the scheduler.
python simplify_onnx.py model-trimmed.onnx
python inference_scheduler.py model-trimmed-simplified.onnx --out-dir /tmp/out
```

This loop is fast because `simplify_onnx.py --check` exits in seconds for
all but the largest models.

---

## 5. Worked example — resnet50-v1-12

The stock model from the ONNX Model Zoo:

```text
resnet50-v1-12.onnx
  175 nodes, opset ai.onnx:12
  input  data: [N, 3, 224, 224]
  output resnetv17_dense0_fwd: [N, 1000]
  ops: Conv×53, BatchNormalization×53, Relu×49, Add×16,
       MaxPool×1, GlobalAveragePool×1, Flatten×1, Gemm×1
```

`BatchNormalization` is not in the scheduler's supported set, and the
batch dim is dynamic. One command fixes both:

```bash
python simplify_onnx.py resnet50-v1-12.onnx --batch 1 --check
```

Result:

```text
resnet50-v1-12-simplified.onnx
  122 nodes
  input  data: [1, 3, 224, 224]
  ops: Conv×53, Relu×49, Add×16, MaxPool×1, GlobalAveragePool×1,
       Flatten×1, Gemm×1
```

All 53 BNs were absorbed into the preceding Convs by onnxsim's constant
folding (the BN scale/bias became compile-time constants once the input
shape was pinned), Flatten + Gemm form the classifier head, and the
scheduler decomposes `Gemm` → `MatMul + Add` at load time. The file is
now ready:

```bash
python inference_scheduler.py resnet50-v1-12-simplified.onnx \
    --out-dir /tmp/resnet50_inference
```
