# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""stories_backend.py — story writing with stories15m on the FPGA
(libstories15m.so, doc/plans/CHAT_PLAN.md §27): Karpathy's llama2.c model
trained on TinyStories — a base model without a chat template, which
continues the text it is given.  Python standard library only (ctypes); the
tokenizer is llama_tokenizer.py, the rest is Smollm2Backend's.

POST /v1/completions        the prompt -> its continuation (OpenAI's legacy
                            completions; the server's text_completion objects)
POST /v1/chat/completions   what a chat user means (plan_chat; system / developer
                            / tool messages are ignored — the model follows no
                            instructions):
    a message                 the opening of a new story; the answer continues it
    "Write / tell me a story  a new story from an opening made of the request:
     about X"                 "Once upon a time, there was X"; the answer starts
                              with that opening (kv260.story "new", "opening")
    "continue", "go on",      the previous story (the earlier user and assistant
     "more", "what happened   messages, read the same way) goes on; a message
     next?" ...               starting with "..." goes on with its own text first
                              (kv260.story "continue")
    the last message the      it goes on, as "continue"
     assistant's
  An answer is never empty: a new story writes at least one token, a
  continuation 16 (CHAT_MIN_TOKENS; after a finished story the model would
  start the next one at once), unless the request sets min_tokens.

prepare()  (no FPGA)  <s> + the text's tokens; a story longer than the context
           less the reserve keeps its most recent tokens (kv260.trimmed_tokens);
           an empty text is refused (the logits after <s> alone are never
           computed: position 0 is the precomputed sink).
generate() (FPGA lock) Smollm2Backend's: prefix-cache reuse, sampling, streamed
           deltas; generation ends at <s> (the model starts the next story) or
           </s>; a completion keeps its leading space, a chat answer drops it.

Defaults: temperature 0.8, top_p 0.9 (llama2.c's run.c: 1.0 / 0.9), no top-k,
no repetition penalty and no DRY — names and phrases repeat in these stories by
design; the loop guard stays on.
"""

from __future__ import annotations

import random
import re
import time
from typing import List, Optional, Sequence, Tuple

from chat_backend import BackendError, ChatRequest, Delta
from llama_tokenizer import Tokenizer
from sampler import SamplerParams, make_sampler
from smollm2_backend import Job, Smollm2Backend

MODEL_ID = "stories15m"
STORY_ROLES = ("user", "assistant")

# chat messages that ask for more of the story (after dropping punctuation and
# the filler words of _FILLER): "Continue the story, please." -> "continue"
CONTINUE = frozenset(("continue", "go on", "more", "keep going", "carry on", "next", "and then",
                      "then what", "what next", "what happened next", "what happens next",
                      "tell me more", "what happened then"))
_FILLER = frozenset(("please", "pls", "the", "story", "it", "now", "ok", "okay", "so"))
# "Write a short story about a cat", "Can you tell me a bedtime story, please?"
_ASK = re.compile(
    r"^(?:(?:please|pls)\s*,?\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:write|tell|make|create|give|generate|compose|read|invent)\s+(?:(?:me|us)\s+)?"
    r"(?:(?:a|an|another|one)\s+)?(?:[\w'-]+\s+){0,3}?(?:story|tale)\b"
    r"(?:\s*,?\s*(?P<how>about|of|with|featuring|where|in\s+which)\s+(?P<topic>.*?))?"
    r"(?:\s*,?\s*please)?[\s.!?]*$", re.I)
CHAT_MIN_TOKENS = {"new": 1, "continue": 16}   # a chat answer's min_tokens (the request's wins)
_PLURAL = re.compile(r"^(?:two|three|four|five|six|seven|eight|nine|ten|some|many|these|those)\b", re.I)
_DET = re.compile(r"^(?:a|an|the|one|my|his|her|their|our|your|this|that)\b", re.I)


def story_opening(how: Optional[str], topic: Optional[str]) -> str:
    """A TinyStories opening for "a story about <topic>" (best effort: the
    model takes any text as the start of a story)."""
    topic = (topic or "").strip().rstrip(".!?,;: ")
    if not topic:
        return "Once upon a time,"
    if how and how.lower() not in ("about", "of", "with", "featuring"):     # "where ..."
        return f"Once upon a time, {topic}"
    if _PLURAL.match(topic):
        return f"Once upon a time, there were {topic}"
    if _DET.match(topic) or topic[0].isupper():
        return f"Once upon a time, there was {topic}"
    if topic.lower().endswith("s") and not topic.lower().endswith("ss"):
        return f"Once upon a time, there were {topic}"
    return f"Once upon a time, there was {'an' if topic[0].lower() in 'aeiou' else 'a'} {topic}"


def classify(text: str) -> Tuple[str, str]:
    """A chat message as ("continue", text to add), ("ask", the opening of a
    requested story) or ("story", the opening of a new story)."""
    t = text.strip()
    if t.startswith(("...", "\u2026")):
        return "continue", t.lstrip(".\u2026").strip()
    words = [w for w in re.sub(r"[^\w\s']", " ", t.lower()).split() if w not in _FILLER]
    if words and " ".join(words) in CONTINUE:
        return "continue", ""
    m = _ASK.match(t)
    if m:
        return "ask", story_opening(m.group("how"), m.group("topic"))
    return "story", t


def _join(a: str, b: str) -> str:
    """a, then b, with a space between where neither has whitespace at the seam."""
    if a and b and not a[-1].isspace() and not b[0].isspace():
        return a + " " + b
    return a + b


def plan_chat(messages: Sequence[dict]) -> Tuple[str, str, str]:
    """(the text to continue, the text the answer starts with, "new" |
    "continue") for a chat conversation (see the module docstring).  Earlier
    turns are read the same way: an answer to a request holds its opening, an
    answer to a story opening follows it, an answer to "continue" extends the
    story before it."""
    msgs = [m for m in messages if m.get("role") in STORY_ROLES]
    story, pending = "", None
    for m in msgs:
        c = (m.get("content") or "").strip()
        if m["role"] == "user":
            pending = classify(c)
            continue
        kind, text = pending or ("continue", "")
        story = (_join(_join(story, text), c) if kind == "continue" else
                 c if kind == "ask" else _join(text, c))
        pending = None
    kind, text = pending or ("continue", "")
    if kind == "continue":
        return _join(story, text), "", "continue"
    return text, (text if kind == "ask" else ""), "new"


class StoriesBackend(Smollm2Backend):
    model_id = MODEL_ID
    fingerprint = "kv260-stories15m-pow2+sink+p12"
    eog_tokens = ("<s>", "</s>")
    cma_mb = 50.0
    completions = True

    def __init__(self, engine, tokenizer_path: str, *, sampler_lib: Optional[str] = None,
                 defaults: Optional[SamplerParams] = None, context_size: int = 256,
                 reserve: int = 128, model_id: Optional[str] = None,
                 cma_mb: Optional[float] = None, loop_guard: bool = True):
        super().__init__(engine, tokenizer_path, sampler_lib=sampler_lib,
                         defaults=defaults or SamplerParams(temperature=0.8, top_p=0.9, top_k=0,
                                                            repetition_penalty=1.0,
                                                            dry_multiplier=0.0),
                         context_size=context_size, reserve=reserve, repeat_last_n=64,
                         cma_mb=cma_mb, loop_guard=loop_guard, model_id=model_id,
                         default_model_id=MODEL_ID)

    def load_host(self) -> None:
        if self.tok is not None:
            return
        t0 = time.monotonic()
        tok = Tokenizer(self.tokenizer_path)
        if self.lib_vocab:
            self._check_vocab(self.lib_vocab, tok)
        self.eog = self._eog_ids(tok)
        self.tok = tok
        self.sampler = make_sampler(self.tok.vocab_size, self.sampler_lib)
        self.bos = self.tok.bos_id
        self._set_breakers(self.dry_breakers)
        self.host_load_s = time.monotonic() - t0

    def prepare(self, req: ChatRequest) -> Job:
        t0 = time.monotonic()
        if self.tok is None:
            self.load_host()
        params = self.sampler_params(req)
        raw = req.raw or {}
        last_n = self._extra(raw, "repeat_last_n", -1, None, integer=True)
        last_n = self.repeat_last_n if last_n is None else last_n
        dry_n = self._extra(raw, "dry_penalty_last_n", -1, None, integer=True)
        dry_n = self.dry_last_n if dry_n is None else dry_n
        breakers = self._breakers_field(raw)
        lg = raw.get("loop_guard")
        if lg is not None and not isinstance(lg, bool):
            raise BackendError("Invalid type for 'loop_guard': expected a boolean.", "loop_guard",
                               code="invalid_type")
        if req.prompt is not None:
            param, text, lead, kind = "prompt", req.prompt, "", None
        else:
            param = "messages"
            text, lead, kind = plan_chat(req.messages)
        body: List[int] = self.tok.encode(text, add_special_tokens=False)
        if not body:
            raise BackendError(
                f"The prompt is empty: {self.model_id} continues a text — give it the beginning of a "
                f"story." if param == "prompt" else
                f"There is no story to continue: {self.model_id} writes children's stories — type "
                f"how one begins (\"Lily found a little red ball.\"), or ask for one (\"Write a story "
                f"about a cat\").", param)
        reserve = min(req.max_tokens, self.reserve) if req.max_tokens else self.reserve
        keep = max(1, self.ctx - reserve - 1)                  # position 0 is the sink <s>
        dropped = max(0, len(body) - keep)
        ids = [self.bos] + body[dropped:]
        seed = req.seed if req.seed is not None else random.SystemRandom().getrandbits(63)
        job = Job(req, ids, params, seed, req.max_tokens, list(req.stop), last_n, dry_n,
                  breakers, self.loop_guard if lg is None else lg, 0,
                  (time.monotonic() - t0) * 1000.0)
        job.extra.update(trimmed_tokens=dropped, lead=lead, story=kind)
        job.extra["top_logprobs"] = self._logprobs_fields(req)
        mt = self._extra(raw, "min_tokens", 0, None, integer=True)
        job.extra["min_tokens"] = mt if mt is not None else CHAT_MIN_TOKENS.get(kind, 0)
        return job

    def _decoder(self, job: Job):
        # a completion keeps the continuation's leading space, as does a chat answer
        # after an opening it repeats; any other chat answer is a message of its own
        strip = job.req.prompt is None and not job.extra.get("lead")
        return self.tok.incremental_decoder(strip_first=strip)

    def generate(self, job: Job, cancel):
        lead = job.extra.get("lead")
        if lead:
            yield Delta(lead)
        for ev in super().generate(job, cancel):
            if hasattr(ev, "info"):                             # the Finish
                if job.extra.get("story"):
                    ev.info["story"] = job.extra["story"]
                    ev.info.setdefault("log", {})["story"] = job.extra["story"]
                if lead:
                    ev.info["opening"] = lead
                if job.extra.get("trimmed_tokens"):
                    ev.info["trimmed_tokens"] = job.extra["trimmed_tokens"]
                    ev.info.setdefault("log", {})["trimmed"] = f"{job.extra['trimmed_tokens']}tok"
            yield ev
