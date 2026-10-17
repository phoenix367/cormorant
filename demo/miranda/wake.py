"""wake.py — the key phrase in a transcript ("Hello Miranda"), fuzzily.

The speech recognizer writes the phrase in many ways — "Hello, Miranda.",
"hello miranda!", "Hallo Maranda", "Hello Mi randa", "Haloma Randa" — so the
phrase is found, anywhere in the transcript, in either of two ways:

  words     each of the phrase's words matched by similarity (difflib's ratio
            >= `ratio`; a recognizer word may also be two words run together);
  sound     one to four transcript words whose consonants, in order, are the
            phrase's (vowels, h, w and y dropped, doubled letters merged,
            voiced / voiceless pairs folded: "Haloma Randa" and "Hello Miranda"
            are both l-m-r-n-t) and whose letters are still similar
            (ratio >= `sound_ratio`) — the vowels are what a recognizer gets
            wrong; "Hello Rhonda" (l-r-n-t) is not the phrase.

  name      at the start of the transcript (after at most two short words —
            the greeting, which the recognizer mangles most: "How old,
            Miranda", "Oh, Mjolanda", "How woe, meogandha"), a word shaped like
            the phrase's last word, the name: letters similar (ratio >=
            `name_ratio`), the same first consonant, its consonants the name's
            with at most one missing ("Muganda" m-k-n-t against m-r-n-t).  A
            name like it at the start of a sentence — "Hello Amanda" — wakes
            Miranda too.

The rest of the transcript after the phrase is the request, in its own
spelling.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

WORD = re.compile(r"[A-Za-z0-9']+")


def _norm(w: str) -> str:
    return w.lower().replace("'", "")


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


FOLD = str.maketrans("bdgvzcqx", "ptkfskkk")


def skeleton(letters: str) -> str:
    """The consonants in order: vowels and h / w / y dropped, voiced and voiceless
    pairs folded (b-p, d-t, g-k, v-f, z-s; c, q, x -> k), repeats merged."""
    out = []
    for ch in letters.lower().translate(FOLD):
        if ch.isalpha() and ch not in "aeiouhwy" and (not out or out[-1] != ch):
            out.append(ch)
    return "".join(out)


def _lcs(a: str, b: str) -> int:
    """The longest common subsequence's length."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


@dataclass
class Match:
    start: int                 # character offsets of the phrase in the transcript
    end: int
    rest: str                  # the transcript after the phrase, stripped of leading punctuation
    score: float


class WakePhrase:
    def __init__(self, phrase: str = "hello miranda", ratio: float = 0.8, sound_ratio: float = 0.6,
                 name_ratio: float = 0.6, lead_words: int = 2, lead_len: int = 6):
        self.words = [_norm(w) for w in WORD.findall(phrase)]
        if not self.words:
            raise ValueError("an empty key phrase")
        self.phrase, self.ratio, self.sound_ratio = phrase, ratio, sound_ratio
        self.letters = "".join(self.words)
        self.skel = skeleton(self.letters)
        self.name = self.words[-1]
        self.name_skel = skeleton(self.name)
        self.name_ratio, self.lead_words, self.lead_len = name_ratio, lead_words, lead_len

    def _name_like(self, w: str) -> Optional[float]:
        """Similarity of a word to the name when it is shaped like it, else None."""
        if not (len(self.name) - 3 <= len(w) <= len(self.name) + 3) or len(self.name_skel) < 2:
            return None
        s = _sim(w, self.name)
        sk = skeleton(w)
        if s < self.name_ratio or not sk or sk[0] != self.name_skel[0] or sk[-1] != self.name_skel[-1]:
            return None
        return s if _lcs(sk, self.name_skel) >= len(self.name_skel) - 1 else None

    def _word(self, words: List[Tuple[str, int, int]], i: int, target: str) -> Optional[Tuple[float, int]]:
        """(similarity, words used) of target at words[i]: one word, or two run together."""
        best = None
        for k in (1, 2):
            if i + k > len(words):
                break
            s = _sim("".join(w for w, _, _ in words[i:i + k]), target)
            if s >= self.ratio and (best is None or s > best[0]):
                best = (s, k)
        return best

    def _sound(self, words: List[Tuple[str, int, int]], i: int) -> Optional[Tuple[float, int]]:
        """(letter similarity, words used) of the phrase by its consonants at words[i]."""
        best = None
        for k in range(1, 5):
            if i + k > len(words):
                break
            cand = "".join(w for w, _, _ in words[i:i + k])
            if skeleton(cand) == self.skel:
                s = _sim(cand, self.letters)
                if s >= self.sound_ratio and (best is None or s > best[0]):
                    best = (s, k)
        return best

    def find(self, text: str) -> Optional[Match]:
        """The first occurrence of the phrase in text, or None."""
        words = [(_norm(m.group()), m.start(), m.end()) for m in WORD.finditer(text)]
        for i in range(len(words)):
            j, scores = i, []
            for target in self.words:
                hit = self._word(words, j, target) if j < len(words) else None
                if hit is None:
                    break
                scores.append(hit[0])
                j += hit[1]
            else:
                return self._match(text, words, i, j, sum(scores) / len(scores))
            hit = self._sound(words, i)
            if hit is not None:
                return self._match(text, words, i, i + hit[1], hit[0])
        for i in range(min(self.lead_words + 1, len(words))):          # the name near the start
            if any(len(w) > self.lead_len for w, _, _ in words[:i]):
                break
            for k in (1, 2):                                            # "Mi randa": two words run together
                if i + k <= len(words):
                    s = self._name_like("".join(w for w, _, _ in words[i:i + k]))
                    if s is not None:
                        return self._match(text, words, 0, i + k, s)
        return None

    @staticmethod
    def _match(text, words, i, j, score) -> Match:
        start, end = words[i][1], words[j - 1][2]
        rest = text[end:].lstrip(" \t\n,.;:!?-—–\"'")
        return Match(start, end, rest.strip(), score)
