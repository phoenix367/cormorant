#!/usr/bin/env python3
"""End-to-end check of the smollm2 backend on the host (CHAT_PLAN phase 4,
gate 2): kv260_chat_server.py with `--backend smollm2 --llm-fake float`
(tests/fake_llm.py's FloatModelEngine: llm_study.py's numpy float64 Llama in
place of libsmollm2.so) is started locally, `chat.py` asks it one-shot and
multi-turn questions greedily (temperature 0), and every answer must be
the text transformers' generate(do_sample=False) produces for the same
conversation (float32 torch model, the tokenizer.json pipeline).

The multi-turn session goes through chat.py's REPL, so the prefix cache is
exercised: each turn prefills only the new tokens (checked in the server
log), and the reference conversation carries the server's earlier answers.

Usage (.venv-export: torch, transformers, numpy):
  /home/ivan/projects/axi_demo/.venv-export/bin/python demo/chat/scripts/e2e_check.py [--max-tokens 96]
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
ASSETS = os.environ.get("SMOLLM_ASSETS", os.path.join(CHAT, "assets", "smollm2-135m-instruct"))
sys.path.insert(0, CHAT)

ONE_SHOT = [
    (None, "What is the capital of France?"),
    (None, "Write a Python function that checks whether a number is prime."),
    (None, "Explain why the sky is blue in simple terms."),
    ("You are a concise assistant. Answer in one sentence.", "Why do leaves change color in autumn?"),
    (None, "Translate 'Good morning, how are you?' into French."),
    (None, "Write a haiku about the ocean 🌊."),
]
MULTI = ["I'm planning a weekend trip to the mountains.", "What should I pack?",
         "Thanks! Any tips for staying safe on the trail?"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count())
    hf_tok = transformers.TokenizersBackend.from_pretrained(ASSETS)
    hf = AutoModelForCausalLM.from_pretrained(ASSETS, dtype=torch.float32, attn_implementation="eager").eval()

    def reference(messages):
        ids = hf_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if not isinstance(ids, list) else ids)
        with torch.no_grad():
            out = hf.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                              max_new_tokens=args.max_tokens, do_sample=False, eos_token_id=2,
                              pad_token_id=2)
        return hf_tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)

    sampler_lib = None
    try:
        from sampler import build_library
        sampler_lib = build_library(os.path.join(tempfile.mkdtemp(prefix="e2e_"), "libsampler.so"))
    except Exception as e:                                        # noqa: BLE001
        print(f"(no C sampler: {e}; the server uses the Python one)")
    port = free_port()
    url = f"http://127.0.0.1:{port}/v1"
    log_path = os.path.join(tempfile.mkdtemp(prefix="e2e_"), "server.log")
    cmd = [sys.executable, os.path.join(CHAT, "kv260_chat_server.py"), "--host", "127.0.0.1",
           "--port", str(port), "--backend", "smollm2", "--llm-fake", "float",
           "--llm-tokenizer", os.path.join(ASSETS, "tokenizer.json")]
    if sampler_lib:
        cmd += ["--llm-sampler-lib", sampler_lib]
    logf = open(log_path, "w")
    srv = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    ok = total = 0
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.load(r)["status"] == "ok":
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit("the server did not come up:\n" + open(log_path).read())
        print(f"server up in {time.time() - t0:.1f} s ({url}, float fake)")
        chat = [sys.executable, os.path.join(CHAT, "chat.py"), "--url", url, "--temperature", "0",
                "--max-tokens", str(args.max_tokens)]

        def show(ours, ref, same):
            print(f"  {'IDENTICAL' if same else 'DIFFERENT'} ({len(ours)} chars)")
            for tag, txt in (("server", ours), ("hf", ref)) if not same else (("answer", ours),):
                print(f"    {tag}: " + txt[:300].replace("\n", "\n            "))

        for system, q in ONE_SHOT:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
            argv = chat + (["--system", system] if system else []) + ["-q", q]
            t = time.time()
            out = subprocess.run(argv, capture_output=True, text=True, timeout=600)
            ours = out.stdout[:-1] if out.stdout.endswith("\n") else out.stdout
            ref = reference(msgs)
            same = ours == ref
            ok += same
            total += 1
            print(f"[{total}] {q!r} ({time.time() - t:.1f} s)")
            show(ours, ref, same)

        # multi-turn through the REPL (history resent by chat.py -> prefix reuse)
        stdin = "".join(line + "\n" for line in MULTI) + "/quit\n"
        out = subprocess.run(chat, input=stdin, capture_output=True, text=True, timeout=1200).stdout
        answers = []
        rest = out
        for i, line in enumerate(MULTI):
            head = f"> {line}\n"
            rest = rest[rest.index(head) + len(head):]
            nxt = f"> {MULTI[i + 1]}\n" if i + 1 < len(MULTI) else "> /quit\n"
            answers.append(rest[:rest.index(nxt)][:-1])
        history = []
        for line, ours in zip(MULTI, answers):
            history.append({"role": "user", "content": line})
            ref = reference(history)
            same = ours == ref
            ok += same
            total += 1
            print(f"[{total}] multi-turn {line!r}")
            show(ours, ref, same)
            history.append({"role": "assistant", "content": ours})
    finally:
        srv.terminate()
        srv.wait(30)
        logf.close()
    log = open(log_path).read()
    reqs = [ln for ln in log.splitlines() if "POST /v1/chat/completions" in ln]
    print("\nserver log (per request):")
    for ln in reqs:
        print("  " + re.sub(r"^\S+ \S+ ", "", ln))
    multi = reqs[-len(MULTI):]
    reuse = [re.search(r"reuse=(\d+)/(\d+) prefill=(\d+)tok", ln) for ln in multi]
    print(f"\n{ok}/{total} answers identical to transformers generate(do_sample=False)")
    grew = all(int(m.group(1)) > 1 for m in reuse[1:])
    print(f"multi-turn prefix reuse: {[f'{m.group(1)}/{m.group(2)} cached, {m.group(3)} prefilled' for m in reuse]}"
          f" -> {'OK' if grew else 'NOT REUSED'}")
    return 0 if ok == total and grew else 1


if __name__ == "__main__":
    sys.exit(main())
