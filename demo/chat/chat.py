#!/usr/bin/env python3
"""
chat.py — a small OpenAI-compatible chat client (Python standard library
only) for the KV260 chat server; runs on a laptop or on the board.

  chat.py [--url http://kv260:8000/v1] [--api-key KEY] [--model M]
          [--doc FILE | --system TEXT] [-q QUESTION] [--no-stream]
          [--max-tokens N] [--temperature T]

Interactive commands:
  /doc FILE      use FILE as the document (the system message; bert-squad)
  /system TEXT   set the system message (/system alone clears it)
  /model [M]     list the server's models / switch to M
  /reset         forget the conversation (keeps the system message)
  /help, /quit

With -q the question is asked once and the answer printed (exit code 1 on
an error).  Environment: KV260_CHAT_URL, KV260_CHAT_API_KEY (or
OPENAI_API_KEY).
"""

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    import readline  # noqa: F401 — line editing and history for input()
except ImportError:                                           # pragma: no cover
    pass

DIM, RST = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


class ApiError(Exception):
    pass


def call(args, path, body=None):
    """GET (body None) or POST JSON; returns the open response."""
    req = urllib.request.Request(args.url.rstrip("/") + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream, application/json"})
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    try:
        return urllib.request.urlopen(req, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode())["error"]["message"]
        except Exception:                                      # noqa: BLE001
            msg = e.reason
        raise ApiError(f"HTTP {e.code}: {msg}") from None
    except urllib.error.URLError as e:
        raise ApiError(f"cannot reach {args.url}: {e.reason}") from None


def models(args):
    with call(args, "/models") as r:
        return [m["id"] for m in json.load(r)["data"]]


def ask(args, messages):
    """One completion; prints the answer as it arrives, returns its text."""
    body = {"model": args.model, "messages": messages, "stream": not args.no_stream}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    t0 = time.monotonic()
    text, extra, usage, finish = "", {}, {}, None
    with call(args, "/chat/completions", body) as r:
        if not body["stream"]:
            obj = json.load(r)
            ch = obj["choices"][0]
            text, finish = ch["message"]["content"] or "", ch["finish_reason"]
            extra, usage = obj.get("kv260") or {}, obj.get("usage") or {}
            print(text, end="", flush=True)
        else:
            for line in r:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise ApiError(chunk["error"]["message"])
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        text += piece
                        print(piece, end="", flush=True)
                    finish = ch.get("finish_reason") or finish
                extra = chunk.get("kv260") or extra
                usage = chunk.get("usage") or usage
    print()
    if args.verbose or sys.stdout.isatty():
        info = [f"{time.monotonic() - t0:.1f} s"]
        if "windows" in extra:
            info.append(f"{extra['windows']}/{extra.get('windows_total', extra['windows'])} window(s)")
        if extra.get("truncated"):
            info.append("document truncated: the server's --max-windows")
        if "confidence" in extra:
            info.append(f"confidence {extra['confidence']:.2f}")
        if usage:
            info.append(f"{usage.get('prompt_tokens')} + {usage.get('completion_tokens')} tokens")
        if finish and finish != "stop":
            info.append(f"finish {finish}")
        print(f"{DIM}[{' · '.join(info)}]{RST}")
    return text


def read_file(path):
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return f.read().strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("KV260_CHAT_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("KV260_CHAT_API_KEY")
                    or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=None, help="default: the server's first model")
    ap.add_argument("--doc", help="document file (sent as the system message)")
    ap.add_argument("--system", help="system message")
    ap.add_argument("-q", "--question", help="ask once and exit")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("-v", "--verbose", action="store_true", help="print timing / token info")
    args = ap.parse_args(argv)

    try:
        system = read_file(args.doc) if args.doc else args.system
        if not args.model:
            args.model = models(args)[0]
    except (ApiError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    history = []

    def convo():
        return ([{"role": "system", "content": system}] if system else []) + history

    if args.question:
        try:
            ask(args, convo() + [{"role": "user", "content": args.question}])
            return 0
        except (ApiError, OSError, http.client.HTTPException, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    tty = sys.stdin.isatty()
    print(f"{DIM}{args.url} · model {args.model}"
          f"{' · document ' + str(len(system.split())) + ' words' if system else ''}"
          f" · /help for commands{RST}")
    while True:
        try:
            line = input("> " if tty else "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not tty and line:
            print(f"> {line}")                  # scripted session: echo the input
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        try:
            if cmd in ("/quit", "/exit"):
                return 0
            elif cmd == "/help":
                print(__doc__[__doc__.index("Interactive"):__doc__.index("With -q")].rstrip())
            elif cmd == "/reset":
                history.clear()
                print(f"{DIM}(conversation cleared){RST}")
            elif cmd == "/system":
                system = arg or None
                print(f"{DIM}(system message {'set' if system else 'cleared'}){RST}")
            elif cmd == "/doc":
                system = read_file(arg)
                history.clear()
                print(f"{DIM}(document: {arg}, {len(system.split())} words; conversation cleared){RST}")
            elif cmd == "/model":
                if arg:
                    args.model = arg
                print(f"{DIM}models: {', '.join(models(args))}; using {args.model}{RST}")
            elif cmd.startswith("/"):
                print(f"{DIM}unknown command {cmd} (/help){RST}")
            else:
                history.append({"role": "user", "content": line})
                history.append({"role": "assistant", "content": ask(args, convo())})
        except ApiError as e:
            if history and history[-1]["role"] == "user":
                history.pop()
            print(f"error: {e}")
        except (OSError, http.client.HTTPException, ValueError) as e:
            if history and history[-1]["role"] == "user":
                history.pop()
            print(f"error: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
