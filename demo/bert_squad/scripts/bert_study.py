#!/usr/bin/env python3
"""BERT-SQuAD numeric feasibility study for the KV260 ap_fixed<16,8> datapath.

Runs bertsquad-12 three ways on SQuAD 1.1 dev examples that fit one 256-token
window:
  float   — numpy interpreter in float32 (validated against onnxruntime)
  q88     — hardware emulation: every tensor that would live in DDR is
            ap_fixed<16,8>; kernel ops (MatMul/Gemm, elementwise Add/Mul) take
            Q8.8 inputs, accumulate exactly, floor + saturate on output (AP_TRN,
            AP_SAT); host CPU regions (embeddings, LayerNorm, GELU, Softmax, mask
            prep) compute in float and round-to-nearest + saturate on output.
  sched   — the partition the phase-1 inference scheduler generates (host
            regions: fused LayerNorm / GELU, Softmax, Gather / OneHot / Cast;
            the word-embedding table is Q8.8 in DDR; everything else on the
            kernels).  bert_sched_check.py compares it bit for bit with the
            scheduler's own simulation.
  variants selected with --policies (see POLS in main()).

Assets (not in git): demo/bert_squad/assets/vocab.txt (bert-base-uncased) and
dev-v1.1.json (SQuAD 1.1 dev).  Run with inference-scheduler/.venv/bin/python.
Reports EM/F1 vs ground truth, agreement with float, per-tensor ranges.
"""
import argparse, collections, json, math, re, sys, time
import numpy as np
import onnx
from onnx import numpy_helper

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.environ.get("BERT_SQUAD_ASSETS",                          # vocab.txt, dev-v1.1.json
                      os.path.join(REPO, "demo", "bert_squad", "assets"))
MODEL = os.environ.get("BERT_SQUAD_MODEL",
                       os.path.join(REPO, "inference-scheduler", "bertsquad-12-simplified.onnx"))

# ----------------------------------------------------------------- text side
# Tokenizer, SQuAD features, span decoding and metrics live in the stdlib-only
# squad_text.py (shared with the KV260 chat server); re-exported here so that
# bert_study.Tokenizer / build_feature / best_span / em / f1 keep working.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import squad_text  # noqa: E402
from squad_text import (Tokenizer, _is_ctrl, _is_punct, _is_ws, _norm,  # noqa: E402,F401
                        best_span, em, f1, whitespace_tokens)

SEQ = squad_text.SEQ


def build_feature(tok, question, context):
    """squad_text.build_feature with ids / seg / mask as int64 [1, SEQ]
    arrays (the ONNX feeds); None when question + context need more than
    one window."""
    f = squad_text.build_feature(tok, question, context, SEQ)
    if f is None:
        return None
    return dict(f, ids=np.array([f["ids"]], np.int64), seg=np.array([f["seg"]], np.int64),
                mask=np.array([f["mask"]], np.int64))

# ----------------------------------------------------------------- quantisers
FRAC = 8
LO, HI = -32768, 32767

def q_trn(x):   # kernel output: AP_TRN (floor) + AP_SAT
    return np.clip(np.floor(np.asarray(x, np.float64) * 256.0), LO, HI) / 256.0

def q_rnd(x):   # host CPU write-back / weight encoding: round + saturate
    return np.clip(np.round(np.asarray(x, np.float64) * 256.0), LO, HI) / 256.0

# ----------------------------------------------------------------- regions
def region_of_sched(n):
    """Partition of the phase-1 inference scheduler (policy "sched"), derived
    independently from the ONNX graph: host CPU regions compute in float and
    round half-to-even + saturate on write-back — the fused LayerNorm and
    GELU subgraphs, Softmax, and the Gather / OneHot / Cast host ops; every
    other arithmetic op (Gemm = MatMul + bias Add, MatMul incl. the
    OneHot x token-type MatMul, embedding / residual / mask Adds, the mask
    Sub / Mul incl. ones x mask, the 1/8 scale Mul) runs on the PL kernels
    (floor + saturate); Reshape / Transpose / Split / Squeeze / Identity move
    data without changing it."""
    name, op = n.name, n.op_type
    if "/LayerNorm/" in name:
        return "cpu:ln:" + name.split("/LayerNorm/")[0]
    if "/intermediate/dense/" in name and op != "Gemm":
        return "cpu:gelu:" + name.split("/intermediate/dense/")[0]
    if op in ("Softmax", "Gather", "OneHot", "Cast"):
        return f"cpu:{op.lower()}:" + name
    if op in ("Identity", "Split", "Squeeze", "Unsqueeze", "Reshape", "Transpose"):
        return "move"
    return "kernel"


def region_of(n, pol):
    name, op = n.name, n.op_type
    if pol.get("sched"):
        return region_of_sched(n)
    if name.startswith("bert/embeddings/"):
        return "cpu:emb"
    if "/LayerNorm/" in name:
        return "cpu:ln:" + name.split("/LayerNorm/")[0]
    if pol.get("fuse_residual_ln") and op == "Add" and re.search(r"layer_\d+/(attention/output|output)/add$", name):
        return "cpu:ln:" + name.rsplit("/add", 1)[0]
    if pol.get("fuse_mask_softmax") and op in ("Add", "Mul") and re.search(r"attention/self/(add|Mul)$", name):
        return "cpu:softmax:" + name.rsplit("/", 1)[0]
    if "/intermediate/dense/" in name and op != "Gemm":
        return "cpu:gelu:" + name.split("/intermediate/dense/")[0]
    if op == "Softmax":
        return "cpu:softmax:" + name.rsplit("/", 1)[0]
    if name in ("bert/encoder/Reshape", "bert/encoder/Cast", "bert/encoder/mul") or \
       name.endswith("attention/self/ExpandDims") or name.endswith("attention/self/sub") or \
       name.endswith("attention/self/mul_1"):
        return "cpu:mask"
    if op in ("Identity", "Cast", "Split", "Squeeze", "Unsqueeze", "Reshape", "Transpose"):
        return "move"          # value-preserving data movement
    return "kernel"            # Gemm, MatMul, Add, Mul, Sub on the PL kernels

# ----------------------------------------------------------------- interpreter
_libm_tanh = np.vectorize(math.tanh, otypes=[np.float64])
_libm_exp = np.vectorize(math.exp, otypes=[np.float64])


class Bert:
    def __init__(self, path, pol=None, base=None):
        pol = pol or {}
        self.pol = pol
        if base is None:
            m = onnx.load(path)
            self.g = m.graph
            self.init0 = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
            self.nodes = list(self.g.node)
        else:
            self.g, self.init0, self.nodes = base.g, base.init0, base.nodes
        self.init = dict(self.init0)
        if pol.get("fold_qscale"):
            for k in list(self.init):
                if re.search(r"attention/self/query/(kernel|bias):0$", k):
                    self.init[k] = self.init[k] * np.float32(0.125)
            for n in self.nodes:
                if re.search(r"attention/self/Mul$", n.name):
                    self.init[n.input[1]] = np.float32(1.0)
        self.region = {id(n): region_of(n, pol) for n in self.nodes}
        consumers = collections.defaultdict(list)
        for n in self.nodes:
            for i in n.input: consumers[i].append(n)
        self.consumers = consumers
        self.outputs = {o.name for o in self.g.output}
        # constants read by PL kernels are Data_t weights; host-CPU regions keep float parameters
        kernel_inputs = {i for n in self.nodes if self.region[id(n)] == "kernel" for i in n.input}
        if pol.get("sched"):
            # the scheduler keeps the word-embedding table in DDR as Data_t (Q8.8);
            # the host Gather copies its rows
            kernel_inputs |= {n.input[0] for n in self.nodes if n.op_type == "Gather"}
        mm_weights = {n.input[1] for n in self.nodes if n.op_type in ("Gemm", "MatMul") and n.input[1] in self.init}
        def qw(k, v):
            if v.dtype.kind != "f" or k not in kernel_inputs:
                return v
            if pol.get("float_weights") and k in mm_weights:
                return v.astype(np.float64)
            if pol.get("w_shift") and k in mm_weights and v.ndim == 2:
                sh = int(np.clip(np.floor(np.log2(127.99 / max(np.abs(v).max(), 1e-9))), 0, 7))
                sc = 256.0 * (1 << sh)
                return np.clip(np.round(v.astype(np.float64) * sc), LO, HI) / sc
            return q_rnd(v)
        self.qinit = {k: qw(k, v) for k, v in self.init.items()}

    def run(self, feeds, mode="float", policy=None, stats=None):
        q = mode != "float"
        env = dict(self.qinit if q else self.init)
        env.update(feeds)
        policy = policy or {}
        for n in self.nodes:
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            ins = [env[i] if i else None for i in n.input]
            op = n.op_type
            reg = self.region[id(n)]
            if op == "Gemm":
                A, B = ins[0], ins[1]
                if a.get("transA", 0): A = A.T
                if a.get("transB", 0): B = B.T
                if q:
                    acc = np.asarray(A, np.float64) @ np.asarray(B, np.float64)
                    if stats is not None: stats["acc_max"] = max(stats.get("acc_max", 0), float(np.abs(acc).max()))
                    y = q_trn(acc)
                    if len(ins) > 2 and ins[2] is not None:
                        y = q_trn(y + ins[2])          # VectorOP bias add
                else:
                    y = A @ B
                    if len(ins) > 2 and ins[2] is not None: y = y + ins[2]
                outs = [y]
            elif op == "MatMul":
                if q and reg == "kernel":
                    acc = np.matmul(np.asarray(ins[0], np.float64), np.asarray(ins[1], np.float64))
                    if stats is not None: stats["acc_max"] = max(stats.get("acc_max", 0), float(np.abs(acc).max()))
                    outs = [acc]
                else:
                    outs = [np.matmul(ins[0], ins[1])]
            elif op in ("Add", "Sub", "Mul", "Div", "Pow"):
                x, y = ins
                f = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide, "Pow": np.power}[op]
                outs = [f(x, y)]
            elif op == "Sqrt": outs = [np.sqrt(ins[0])]
            elif op == "Reciprocal": outs = [1.0 / ins[0]]
            elif op == "Tanh":
                # policy "sched": the host C calls libm's tanh (numpy's SIMD tanh
                # differs by up to 3 ulp); Python's math module is that libm
                outs = [_libm_tanh(ins[0]) if self.pol.get("sched") else np.tanh(ins[0])]
            elif op == "ReduceMean":
                outs = [np.mean(ins[0], axis=tuple(a["axes"]), keepdims=bool(a.get("keepdims", 1)))]
            elif op == "Softmax":
                x = ins[0]; ax = a.get("axis", 1)
                xm = x - x.max(axis=ax, keepdims=True)
                e = _libm_exp(xm) if self.pol.get("sched") else np.exp(xm)
                outs = [e / e.sum(axis=ax, keepdims=True)]
            elif op == "Reshape":
                shp = [int(s) for s in ins[1]]
                shp = [ins[0].shape[i] if s == 0 else s for i, s in enumerate(shp)]
                outs = [ins[0].reshape(shp)]
            elif op == "Transpose": outs = [np.transpose(ins[0], a["perm"])]
            elif op == "Identity": outs = [ins[0]]
            elif op == "Cast":
                outs = [ins[0].astype({1: np.float32, 7: np.int64, 6: np.int32}[a["to"]])]
            elif op == "Gather": outs = [np.take(ins[0], ins[1], axis=a.get("axis", 0))]
            elif op == "OneHot":
                idx, depth, vals = ins; depth = int(np.asarray(depth).reshape(-1)[0])
                oh = (idx[..., None] == np.arange(depth)).astype(np.float32)
                outs = [oh * (vals[1] - vals[0]) + vals[0]]
            elif op == "Squeeze": outs = [np.squeeze(ins[0], axis=tuple(a["axes"]))]
            elif op == "Unsqueeze": outs = [np.expand_dims(ins[0], axis=tuple(a["axes"]))]
            elif op == "Split":
                ax = a.get("axis", 0); sp = a.get("split")
                if sp is None: outs = np.split(ins[0], len(n.output), axis=ax)
                else: outs = np.split(ins[0], np.cumsum(sp)[:-1], axis=ax)
            else:
                raise NotImplementedError(op)
            for name, v in zip(n.output, outs):
                if q and np.asarray(v).dtype.kind == "f":
                    v = self._quantize(n, name, v, reg, policy, stats)
                elif not q and np.asarray(v).dtype.kind == "f":
                    v = np.asarray(v, np.float32)
                env[name] = v
        return env

    def _quantize(self, n, name, v, reg, policy, stats):
        if reg == "move":
            return v
        leaves = any(self.region[id(c)] != reg for c in self.consumers.get(name, [])) or name in self.outputs
        if reg.startswith("cpu") and not leaves:
            return np.asarray(v, np.float64)        # stays inside a float host region
        # tensor is written to DDR as Data_t
        if stats is not None:
            if reg.startswith("cpu"):
                key = "cpu:" + reg.split(":")[1]
            else:
                tag = (name.split("/")[-2] + "/" + name.split("/")[-1]) if "/" in name else name
                tag = re.sub(r":0$", "", tag)
                tag = re.sub(r"(query|key|value)/BiasAdd", "qkv/BiasAdd", tag)
                key = f"kernel:{n.op_type}:{tag}"
            s = stats.setdefault(key, [0.0, 0, 0])
            s[0] = max(s[0], float(np.abs(v).max())); s[1] += int((np.abs(v) >= 128).sum()); s[2] += v.size
        scale = float(1 << self.pol.get("p_shift", 0)) if reg.startswith("cpu:softmax") else 1.0
        if reg.startswith("cpu"):
            return q_rnd(v * scale) / scale if scale != 1.0 else q_rnd(v)
        return q_trn(v)

# ----------------------------------------------------------------- policies
POLS = {
    "q88":        {},
    "fused":      {"fuse_residual_ln": 1, "fuse_mask_softmax": 1, "fold_qscale": 1},
    "fused+w":    {"fuse_residual_ln": 1, "fuse_mask_softmax": 1, "fold_qscale": 1, "w_shift": 1},
    "fused+w+p7": {"fuse_residual_ln": 1, "fuse_mask_softmax": 1, "fold_qscale": 1, "w_shift": 1, "p_shift": 7},
    "fused+fw":   {"fuse_residual_ln": 1, "fuse_mask_softmax": 1, "fold_qscale": 1, "float_weights": 1},
    "sched":      {"sched": 1},   # exactly the phase-1 scheduler partition (bert_sched_check.py)
}

# ----------------------------------------------------------------- examples
def pick_examples(n):
    """The study's example set: n single-window SQuAD dev questions drawn
    (seed 0) from the first 4n that fit one 256-token window."""
    tok = Tokenizer(f"{HERE}/vocab.txt")
    data = json.load(open(f"{HERE}/dev-v1.1.json"))["data"]
    exs = []
    for art in data:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                f = build_feature(tok, qa["question"], para["context"])
                if f is not None:
                    exs.append((qa, f))
            if len(exs) >= 4 * n: break
        if len(exs) >= 4 * n: break
    rng = np.random.default_rng(0)
    return tok, [exs[i] for i in sorted(rng.choice(len(exs), n, replace=False))]


def feeds_of(f):
    return {"unique_ids_raw_output___9:0": np.array([0], np.int64), "segment_ids:0": f["seg"],
            "input_mask:0": f["mask"], "input_ids:0": f["ids"]}

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--check-ort", action="store_true")
    ap.add_argument("--policies", default="q88,fused,fused+w,fused+w+p7,fused+fw")
    args = ap.parse_args()
    tok, pick = pick_examples(args.n)
    bert = Bert(MODEL)
    if args.check_ort:
        import onnxruntime as ort
        s = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
        qa, f = pick[0]
        feeds = {"unique_ids_raw_output___9:0": np.array([0], np.int64), "segment_ids:0": f["seg"],
                 "input_mask:0": f["mask"], "input_ids:0": f["ids"]}
        r = s.run(["unstack:0", "unstack:1"], feeds)
        env = bert.run(feeds, "float")
        print("ORT vs numpy float max|diff| start %.2e end %.2e" % (
            np.abs(r[0] - env["unstack:0"]).max(), np.abs(r[1] - env["unstack:1"]).max()))
    sel = args.policies.split(",")
    models = {k: Bert(MODEL, POLS[k], base=bert) for k in sel}
    tot = collections.Counter(); stats = {k: {} for k in sel}
    t0 = time.time()
    for k_ex, (qa, f) in enumerate(pick):
        feeds = {"unique_ids_raw_output___9:0": np.array([0], np.int64), "segment_ids:0": f["seg"],
                 "input_mask:0": f["mask"], "input_ids:0": f["ids"]}
        golds = [x["text"] for x in qa["answers"]]
        ef = bert.run(feeds, "float")
        sp_f, txt_f = best_span(f, ef["unstack:0"][0], ef["unstack:1"][0])
        tot["n"] += 1
        tot["em:float"] += max(em(txt_f, g) for g in golds); tot["f1:float"] += max(f1(txt_f, g) for g in golds)
        line = []
        for k in sel:
            eq = models[k].run(feeds, "q88", None, stats[k])
            sp_q, txt_q = best_span(f, eq["unstack:0"][0], eq["unstack:1"][0])
            tot["em:" + k] += max(em(txt_q, g) for g in golds); tot["f1:" + k] += max(f1(txt_q, g) for g in golds)
            tot["same:" + k] += int(sp_q == sp_f)
            lim = f["ctx_off"] + f["n_ctx"]
            tot["err:" + k] += float(max(np.abs(ef["unstack:0"][0] - eq["unstack:0"][0])[:lim].max(),
                                         np.abs(ef["unstack:1"][0] - eq["unstack:1"][0])[:lim].max()))
            line.append(f"{k}={'=' if sp_q == sp_f else 'X'}")
        print(f"[{k_ex:3d}] {' '.join(line)}  {time.time() - t0:.0f}s", flush=True)
    n = tot["n"]
    print(f"\n{n} single-window SQuAD dev examples, {time.time() - t0:.0f} s")
    print(f"{'policy':12s} {'EM':>6s} {'F1':>6s}  same-span  mean max|logit err|")
    print(f"{'float':12s} {100 * tot['em:float'] / n:6.1f} {100 * tot['f1:float'] / n:6.1f}")
    for k in sel:
        print(f"{k:12s} {100 * tot['em:' + k] / n:6.1f} {100 * tot['f1:' + k] / n:6.1f}  {tot['same:' + k]:4d}/{n:<4d}  {tot['err:' + k] / n:.3f}")
    for k in sel:
        print(f"\n[{k}] DDR tensors: max|x| before quantisation, saturated elements")
        for key in sorted(stats[k]):
            if key == "acc_max": continue
            mx, sat, cnt = stats[k][key]
            print(f"  {key:44s} max|x| {mx:9.2f}   sat {sat:9d} / {cnt:11d} ({100 * sat / cnt:.4f} %)")
        print(f"  matmul accumulator max|acc| {stats[k].get('acc_max', 0):.1f}")

if __name__ == "__main__":
    main()
