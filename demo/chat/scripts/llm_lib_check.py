#!/usr/bin/env python3
"""
llm_lib_check.py — exercise libsmollm2.so through ctypes the way the chat
server does (runs ON THE BOARD, Python stdlib only; phase-4 requirements
of doc/CHAT_PLAN.md §12):

  * load the library, check that it exports only llm_* symbols (nm -D) and
    llm_vocab_size() / llm_context_size();
  * llm_open(NULL) (the build's weights dir), CmaFree before / after;
  * a prompt prefilled in one call vs. in several llm_prefill() chunks
    (each returning the logits after its last token) — identical logits;
  * greedy decode steps issued from different threads (calls serialised,
    no thread-local state) — identical to the same steps from one thread;
  * llm_truncate + re-prefill of a second turn;
  * llm_close() (CmaFree back), llm_open() again, the first result again.

usage (on the board): python3 llm_lib_check.py [/root/kv260_chat/lib/libsmollm2.so]
prints one JSON line "LLM_LIB_CHECK: {...}" and exits 0 when every check passed.
"""
import ctypes
import json
import subprocess
import sys
import threading
import time

LIB = sys.argv[1] if len(sys.argv) > 1 else "/root/kv260_chat/lib/libsmollm2.so"
# "<|im_start|>" is the sink (position 0); a short ChatML conversation after it
PROMPT = [9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3398, 4945, 28, 7018, 411, 407,
          19712, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520,
          9531, 198]
TURN2 = [2, 198, 1, 4093, 198, 1780, 314, 2256, 2476, 47, 2, 198, 1, 520, 9531, 198]


def cma_free():
    for line in open("/proc/meminfo"):
        if line.startswith("CmaFree:"):
            return int(line.split()[1])
    return -1


def main():
    res = {"lib": LIB}
    syms = subprocess.run(["nm", "-D", "--defined-only", LIB], capture_output=True, text=True).stdout
    names = sorted(ln.split()[-1] for ln in syms.splitlines() if ln.strip())
    res["exports"] = names
    res["only_llm_exports"] = all(n.startswith("llm_") or n in ("_init", "_fini") for n in names)
    L = ctypes.CDLL(LIB)
    for fn, rt, at in (("llm_open", ctypes.c_int, [ctypes.c_char_p]),
                       ("llm_close", None, []),
                       ("llm_last_error", ctypes.c_char_p, []),
                       ("llm_vocab_size", ctypes.c_int, []),
                       ("llm_context_size", ctypes.c_int, []),
                       ("llm_position", ctypes.c_int, []),
                       ("llm_truncate", ctypes.c_int, [ctypes.c_int]),
                       ("llm_prefill", ctypes.c_int, [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                                      ctypes.POINTER(ctypes.c_float)]),
                       ("llm_decode", ctypes.c_int, [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)])):
        f = getattr(L, fn)
        f.restype = rt
        f.argtypes = at
    V, C = L.llm_vocab_size(), L.llm_context_size()
    res.update(vocab=V, context=C)
    logits = (ctypes.c_float * V)()

    def prefill(tokens):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        rc = L.llm_prefill(arr, len(tokens), logits)
        if rc != 0:
            raise RuntimeError(L.llm_last_error().decode())
        return bytes(logits)

    def argmax(b):
        f = (ctypes.c_float * V).from_buffer_copy(b)
        return max(range(V), key=lambda i: (f[i], -i))

    def decode(tok):
        rc = L.llm_decode(tok, logits)
        if rc != 0:
            raise RuntimeError(L.llm_last_error().decode())
        return bytes(logits)

    c0 = cma_free()
    t0 = time.time()
    rc = L.llm_open(None)
    res["open_s"] = round(time.time() - t0, 3)
    if rc != 0:
        res["error"] = L.llm_last_error().decode()
        print("LLM_LIB_CHECK: " + json.dumps(res))
        return 1
    c1 = cma_free()
    res["cma_used_mb"] = round((c0 - c1) / 1024, 1)
    res["position_after_open"] = L.llm_position()
    one = prefill(PROMPT)
    L.llm_truncate(1)
    chunks = None
    for a, b in ((0, 5), (5, 22), (22, len(PROMPT))):
        chunks = prefill(PROMPT[a:b])
    res["chunked_prefill_identical"] = chunks == one
    res["position_after_prefill"] = L.llm_position()
    # greedy decode, one thread
    toks, seq1 = [], []
    b = one
    for _ in range(4):
        t = argmax(b)
        toks.append(t)
        b = decode(t)
        seq1.append(b)
    # the same steps from four different threads, serialised
    L.llm_truncate(1 + len(PROMPT))
    seq2 = []
    lock = threading.Lock()
    for t in toks:
        th = threading.Thread(target=lambda t=t: (lock.acquire(), seq2.append(decode(t)),
                                                  lock.release()))
        th.start()
        th.join()
    res["threads_identical"] = seq1 == seq2
    res["greedy"] = toks
    # a second turn after truncate
    L.llm_truncate(1 + len(PROMPT) + len(toks))
    b = prefill(TURN2)
    res["turn2_next"] = argmax(b)
    res["position_after_turn2"] = L.llm_position()
    L.llm_close()
    c2 = cma_free()
    res["cma_free_kb"] = {"before_open": c0, "after_open": c1, "after_close": c2}
    rc = L.llm_open(None)
    c3 = cma_free()
    again = prefill(PROMPT) if rc == 0 else b""
    res["reopen_identical"] = again == one
    res["cma_free_kb"]["after_reopen"] = c3
    L.llm_close()
    res["cma_free_kb"]["after_final_close"] = cma_free()
    ok = (res["only_llm_exports"] and res["chunked_prefill_identical"] and res["threads_identical"]
          and res["reopen_identical"] and abs(c2 - c0) < 4096 and V == 49152 and C == 1024)
    res["ok"] = ok
    print("LLM_LIB_CHECK: " + json.dumps(res))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
