"""Step 3c: turn the raw Haiku outputs into the frozen corpus.

  1. punctuation: ; : -> '.', dashes / quotes / brackets / other symbols -> space   (rule 9)
  2. leftover off-list words: one replacement dictionary for the whole corpus, built by Haiku from the word types
     (not per occurrence), each entry checked against the rules       -> data/replacements.json  (cached)
  3. any sentence that still breaks rule 1 or 5 is dropped
  4. chunks are joined back into articles in order                    -> ../corpus/*.json, ../corpus/stats.json

The notebook only reads ../corpus/. Nothing here is needed to run it.
"""
import asyncio, collections, json, re, sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import cavelib, gateway

DATA = Path(__file__).parent / "data"
CORPUS = Path(__file__).parent.parent / "corpus"
BATCH = 150
CONCURRENCY = 16
MAP_PROMPT = ("You help write a closed-vocabulary 'caveman English'. For every input word give a replacement made ONLY of "
              "words from the WORD LIST below (lowercase, base forms, at most 4 words), meaning the same thing as "
              "closely as possible. For a name of a person or place, describe it (e.g. 'Vienna' -> 'big town', "
              "'Coptic' -> 'old church'). If the word adds no meaning, answer with -.\n"
              "Answer one line per input word, in the same order, exactly as: <word> => <replacement>\nNo other text.\n\n"
              "WORD LIST ({n} words):\n{words}")


def fix_punct(text):
    text = re.sub(r"[;:]", ".", text)
    text = re.sub(r"[^\sA-Za-z0-9.,?!]", " ", text)
    text = re.sub(r"\s*([.,?!])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([.,?!])\1+", r"\1", text)
    return "\n\n".join(p.strip() for p in re.split(r"\n\s*\n", text) if p.strip())


async def build_map(words, W):
    system = MAP_PROMPT.format(n=len(W), words=" ".join(sorted(W)))
    out, sem = {}, asyncio.Semaphore(CONCURRENCY)

    async def one(batch):
        async with sem:
            try:
                text, _ = await gateway.chat(client, [{"role": "system", "content": system},
                                                      {"role": "user", "content": "\n".join(batch)}], max_tokens=32000)
            except Exception as e:
                print("map batch failed:", e, flush=True); return
        for line in text.splitlines():
            if "=>" in line:
                w, r = (s.strip() for s in line.split("=>", 1))
                if w in batch:
                    out[w] = "" if r in ("-", "") else r.lower()
        print(f"mapped {len(out):,}/{len(words):,}", flush=True)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(words[i:i + BATCH]) for i in range(0, len(words), BATCH)))
    return out


def apply_map(text, mp):
    return re.sub(r"[A-Za-z]+", lambda m: mp.get(m.group(), m.group()), text)


def drop_bad_sentences(text, W, allowed):
    kept, dropped, paras = 0, 0, []
    for p in text.split("\n\n"):
        sents = re.findall(r"[^.?!]+[.?!]*", p)
        good = []
        for s in sents:
            if not s.strip():
                continue
            if cavelib.check(s, W, allowed)["bad"]:
                dropped += 1
            else:
                good.append(s.strip()); kept += 1
        if good:
            paras.append(" ".join(good))
    return "\n\n".join(paras), kept, dropped


def main():
    W, names = cavelib.load_wordlist(), cavelib.load_names()
    chunks = {c["chunk_id"]: c for c in map(json.loads, open(DATA / "chunks.jsonl"))}
    raw = {r["chunk_id"]: r for r in map(json.loads, open(DATA / "caveman_raw.jsonl"))}
    print(f"raw outputs: {len(raw):,} of {len(chunks):,} chunks")

    texts, bad_types = {}, collections.Counter()
    for cid, r in raw.items():
        allowed = names | set(r["subject"])
        t = fix_punct(cavelib.normalize(r["caveman"], W, allowed))
        texts[cid] = t
        bad_types.update(cavelib.check(t, W, allowed)["bad"])
    n_words = sum(cavelib.check(t, W, names | set(raw[c]["subject"]))["n_words"] for c, t in texts.items())
    print(f"after punctuation fix: {sum(bad_types.values()):,} off-list words ({sum(bad_types.values())/n_words:.2%}), "
          f"{len(bad_types):,} types")

    map_path = DATA / "replacements.json"
    mp = json.loads(map_path.read_text()) if map_path.exists() else {}
    todo = [w for w in bad_types if w not in mp]
    if todo:
        mp.update(asyncio.run(build_map(todo, W)))
        map_path.write_text(json.dumps(mp, indent=0, ensure_ascii=False, sort_keys=True))
    valid = {w: r for w, r in mp.items() if not cavelib.check(r, W, set())["bad"]}
    print(f"replacement dictionary: {len(mp):,} entries, {len(valid):,} valid")

    stats = collections.Counter()
    by_article = collections.defaultdict(list)
    for cid, t in texts.items():
        r = raw[cid]
        allowed = names | set(r["subject"])
        t = apply_map(t, {w: v for w, v in valid.items() if w not in allowed})
        t = re.sub(r"[ \t]+", " ", t)
        t, kept, dropped = drop_bad_sentences(t, W, allowed)
        stats["sent_kept"] += kept; stats["sent_dropped"] += dropped
        by_article[r["article_id"]].append((int(cid.rsplit("_", 1)[1]), t))

    arts = [json.loads(l) for l in open(DATA / "articles.jsonl")]
    CORPUS.mkdir(exist_ok=True)
    out = {"train": [], "held": []}
    src_held, meta = [], []
    for a in arts:
        parts = [t for _, t in sorted(by_article.get(a["id"], [])) if t]
        if not parts:
            continue
        n_chunks = sum(1 for c in chunks.values() if c["article_id"] == a["id"])
        complete = len(by_article[a["id"]]) == n_chunks
        if a["split"] == "held" and not complete:   # held-out docs must be whole, so the English twin matches
            continue
        out[a["split"]].append("\n\n".join(parts))
        if a["split"] == "held":
            src_held.append(a["text"])
        meta.append({"id": a["id"], "title": a["title"], "split": a["split"], "topic": a["topic"],
                     "url": f"https://en.wikipedia.org/?curid={a['id']}", "complete": complete})

    (CORPUS / "caveman_train.json").write_text(json.dumps(out["train"], ensure_ascii=False, indent=0))
    (CORPUS / "caveman_heldout.json").write_text(json.dumps(out["held"], ensure_ascii=False, indent=0))
    (CORPUS / "heldout_source_en.json").write_text(json.dumps(src_held, ensure_ascii=False, indent=0))
    (CORPUS / "articles.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta))
    allw = collections.Counter(w for d in out["train"] for w in re.findall(r"[A-Za-z]+", d))
    s = {"train_docs": len(out["train"]), "train_bytes": sum(len(d.encode()) for d in out["train"]),
         "held_docs": len(out["held"]), "held_bytes": sum(len(d.encode()) for d in out["held"]),
         "held_source_bytes": sum(len(d.encode()) for d in src_held),
         "raw_offlist_rate": sum(bad_types.values()) / n_words, "replacement_entries_valid": len(valid),
         "sentences_kept": stats["sent_kept"], "sentences_dropped": stats["sent_dropped"],
         "train_word_types": len(allw), "train_lowercase_types": sum(1 for w in allw if w.islower()),
         "train_name_types": sum(1 for w in allw if not w.islower())}
    (CORPUS / "stats.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
