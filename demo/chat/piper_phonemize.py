"""piper_phonemize.py — text -> Piper phoneme ids through espeak-ng (ctypes;
Python standard library only; runs on the board with the distribution's
libespeak-ng.so.1 and espeak-ng-data).

What Piper's own phonemizer (piper-phonemize) does, on a stock espeak-ng:
espeak_TextToPhonemes() returns the IPA of one clause at a time and advances
the text pointer past it; the clause's terminator (. , ! ? ; :) is taken from
the text it consumed and appended to its phonemes, the way Piper's espeak
fork reports it.  Sentences end at . ! ?; ids are Piper's: ^ _ (phoneme _)* $
(BOS, pad after BOS and after every phoneme, EOS), characters outside the
voice's phoneme_id_map are dropped.  Language-switch flags such as "(fr)"
are removed.  One Espeak instance per process (espeak-ng is not re-entrant).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

AUDIO_OUTPUT_SYNCHRONOUS = 2
CHARS_UTF8 = 1
PHONEMES_IPA = 0x02
CLAUSE_PUNCT = ".,!?;:"
SENTENCE_END = ".!?"
_FLAGS = re.compile(r"\([a-z]{2,3}(-[a-z0-9]+)?\)")
_LOOKAHEAD = re.compile(r"(?<=\s)[^\s.,!?;:]+$")


class PhonemizeError(Exception):
    pass


class Espeak:
    """espeak-ng in phoneme mode for one voice."""

    _lock = threading.Lock()

    def __init__(self, voice: str = "en-us", lib: Optional[str] = None, data: Optional[str] = None):
        path = lib or ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1"
        try:
            self.lib = ctypes.cdll.LoadLibrary(path)
        except OSError as e:
            raise PhonemizeError(f"cannot load {path}: {e} (apt install libespeak-ng1)") from e
        L = self.lib
        L.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        L.espeak_Initialize.restype = ctypes.c_int
        L.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        L.espeak_SetVoiceByName.restype = ctypes.c_int
        L.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        L.espeak_TextToPhonemes.restype = ctypes.c_char_p
        if L.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data.encode() if data else None, 0) <= 0:
            raise PhonemizeError(f"espeak_Initialize failed (data path {data or 'default'})")
        if L.espeak_SetVoiceByName(voice.encode()) != 0:
            raise PhonemizeError(f"espeak-ng has no voice '{voice}'")
        self.voice = voice

    def clauses(self, text: str) -> List[Tuple[str, str]]:
        """[(IPA phonemes, terminator or '')], one per espeak clause."""
        raw = text.encode("utf-8")
        buf = ctypes.create_string_buffer(raw)
        base = ctypes.addressof(buf)
        ptr = ctypes.c_void_p(base)
        out = []
        with self._lock:
            while ptr.value is not None:
                start = ptr.value
                ph = self.lib.espeak_TextToPhonemes(ctypes.byref(ptr), CHARS_UTF8, PHONEMES_IPA)
                stop = ptr.value if ptr.value is not None else base + len(raw)
                if stop <= start and ptr.value is not None:        # no progress: give up
                    break
                # espeak reads ahead: the pointer stops a character or a word
                # into the next clause ("Hello! I"); drop that partial word
                used = raw[start - base:stop - base].decode("utf-8", "replace").rstrip()
                used = _LOOKAHEAD.sub("", used).rstrip()
                term = used[-1] if used and used[-1] in CLAUSE_PUNCT else ""
                ph = _FLAGS.sub("", (ph or b"").decode("utf-8", "replace")).strip()
                if ph or term:
                    out.append((ph, term))
        return out


def sentences(clauses: Sequence[Tuple[str, str]]) -> List[str]:
    """Phoneme strings, one per sentence: clause phonemes + terminator, joined
    by spaces; a sentence ends after . ! ?; sentences without a phoneme
    letter (only punctuation) are dropped."""
    out, cur = [], []
    for ph, term in clauses:
        cur.append(ph + term)
        if term in SENTENCE_END and term:
            out.append(" ".join(c for c in cur if c))
            cur = []
    if cur:
        out.append(" ".join(c for c in cur if c))
    return [s for s in out if any(ch.isalpha() for ch in s)]      # nothing to say: dropped


def to_ids(phonemes: str, id_map: Dict[str, List[int]]) -> List[int]:
    """Piper's phonemes_to_ids: ^ _ (phoneme _)* $."""
    pad = id_map["_"]
    ids = list(id_map["^"]) + list(pad)
    for ch in phonemes:
        if ch in id_map:
            ids += list(id_map[ch]) + list(pad)
    return ids + list(id_map["$"])


def utterances(sents: Sequence[str], id_map: Dict[str, List[int]], max_ids: int = 400) -> List[List[int]]:
    """Whole sentences packed into utterances of at most max_ids ids (one
    front-end pass each); a longer sentence is split at word boundaries."""
    pieces: List[str] = []
    for s in sents:
        if len(to_ids(s, id_map)) <= max_ids:
            pieces.append(s)
            continue
        for part in _split(s, id_map, max_ids):
            pieces.append(part)
    out, cur = [], ""
    for p in pieces:
        cand = (cur + " " + p) if cur else p
        if cur and len(to_ids(cand, id_map)) > max_ids:
            out.append(to_ids(cur, id_map))
            cur = p
        else:
            cur = cand
    if cur:
        out.append(to_ids(cur, id_map))
    return out


def _split(s: str, id_map, max_ids: int) -> List[str]:
    parts, cur = [], ""
    for tok in s.split():
        cand = (cur + " " + tok) if cur else tok
        if cur and len(to_ids(cand, id_map)) > max_ids:
            parts.append(cur)
            cur = tok
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts
