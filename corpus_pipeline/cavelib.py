"""Shared helpers for building the caveman corpus: prompt, user message, rule checker."""
import json, re
from pathlib import Path

HERE = Path(__file__).parent
MONTHS_DAYS = set("January February March April May June July August September October November December "
                  "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())
NAME_RE = re.compile(r"^[A-Z][a-z]+$")


def load_wordlist():
    return set((HERE / "wordlist.txt").read_text().split())


def load_names():
    p = HERE / "core_names.txt"
    return set(p.read_text().split()) if p.exists() else set()


def subject_names(title, text, wordlist):
    """Capitalized title words that are real names: not list words and (almost) never lowercase in the article.
    'Battle of Hastings' -> ['Hastings'], 'Photosynthesis' -> []."""
    title = re.sub(r"\(.*?\)", " ", title)
    out = []
    for w in re.findall(r"[A-Za-z]+", title):
        if not NAME_RE.match(w) or w.lower() in wordlist or w in out:
            continue
        n_cap = len(re.findall(rf"\b{w}\b", text))
        n_low = len(re.findall(rf"\b{w.lower()}\b", text))
        if n_low <= 0.1 * max(n_cap, 1):
            out.append(w)
    return out


def user_message(title, subject, text):
    return f"TITLE: {title}\nSUBJECT NAMES: {', '.join(subject) or '(none)'}\nTEXT:\n{text}"


def system_prompt(wordlist, names):
    examples = json.loads((HERE / "examples.json").read_text())
    ex = "\n\n".join(user_message(e["title"], e["subject"], e["src"]) + f"\nCAVEMAN:\n{e['cave']}" for e in examples)
    return ((HERE / "style_guide.md").read_text()
            .replace("{examples}", ex)
            .replace("{n_words}", str(len(wordlist))).replace("{wordlist}", " ".join(sorted(wordlist)))
            .replace("{n_names}", str(len(names))).replace("{namelist}", " ".join(sorted(names))))


def normalize(text, wordlist, allowed_names):
    """Deterministic fix before checking: 'This' / 'It' at a sentence start -> 'this' / 'it' (rule 2)."""
    return re.sub(r"\b[A-Z][a-z]*\b",
                  lambda m: m.group().lower() if m.group() not in allowed_names and m.group().lower() in wordlist
                  else m.group(), text)


def check(text, wordlist, allowed_names):
    """Rule check of one caveman output. Lowercase words must be list words, Capitalized words must be allowed
    names; anything else (mixed case, ALLCAPS, unknown word) is a violation."""
    words = re.findall(r"[A-Za-z]+", text)
    bad = []
    for w in words:
        if w.islower():
            ok = w in wordlist
        elif NAME_RE.match(w) or (len(w) == 1 and w.isupper()):
            ok = w in allowed_names
        else:
            ok = False
        if not ok:
            bad.append(w)
    bad_punct = re.findall(r"[^\sA-Za-z0-9.,?!]", text)
    n = max(len(words), 1)
    return {"n_words": len(words), "bad": bad, "oov_rate": len(bad) / n, "bad_punct": len(bad_punct)}
