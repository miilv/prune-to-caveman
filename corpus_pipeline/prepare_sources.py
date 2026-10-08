"""Step 3a: build the source side of the caveman corpus.

  python prepare_sources.py candidates   # clean articles from seeded Wikipedia shards -> data/candidates.jsonl
  (python classify_topics.py)            # topic label per article             -> data/topics.jsonl
  python prepare_sources.py split        # keep on-topic articles, split train / held-out BY ARTICLE, build the
                                         # core-name list from train sources only, cut into request-sized chunks
                                         # -> data/articles.jsonl, data/chunks.jsonl, core_names.txt
"""
import collections, json, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import cavelib

SEED = 0
N_SHARDS = 2
TRAIN_SRC_BYTES = 14_000_000      # English source; caveman output is shorter (measured in the pilot)
HELD_SRC_BYTES = 1_200_000
CHUNK_CHARS = 6000
MIN_DF = 50                       # core name = capitalized in >= 50 train articles
OUT = Path(__file__).parent / "data"
STOP_SECTIONS = {"See also", "References", "External links", "Notes", "Further reading", "Bibliography", "Sources",
                 "Footnotes", "Citations", "Notes and references", "Works cited", "Gallery"}


def clean(text):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s in STOP_SECTIONS:
            break
        if not s or (len(s) < 60 and not s.endswith((".", "!", "?"))):   # headings, list fragments
            continue
        out.append(s)
    return "\n\n".join(out)


def keep_article(title, text):
    if title.startswith(("List of", "Index of", "Lists of")) or not 3000 <= len(text) <= 40000:
        return False
    non_ascii = sum(ord(c) > 127 for c in text) / len(text)
    return non_ascii < 0.03


def chunks(text):
    cur, out = [], []
    for p in text.split("\n\n"):
        if cur and sum(len(x) for x in cur) + len(p) > CHUNK_CHARS:
            out.append("\n\n".join(cur)); cur = []
        cur.append(p)
    if cur:
        out.append("\n\n".join(cur))
    return out


def core_names(train_articles, wordlist):
    """Names = Capitalized words not at a sentence start, that are (almost) never lowercase in the corpus."""
    df, cap, low = collections.Counter(), collections.Counter(), collections.Counter()
    for a in train_articles:
        seen = set()
        for m in re.finditer(r"\b[A-Za-z]+\b", a["text"]):
            w = m.group()
            if w.islower():
                low[w] += 1
                continue
            if not cavelib.NAME_RE.match(w):
                continue
            before = a["text"][max(0, m.start() - 2):m.start()]
            if m.start() == 0 or re.search(r"[.!?\n]\s?$|^\n", before):   # sentence-initial -> not evidence
                continue
            cap[w] += 1
            seen.add(w)
        df.update(seen)
    names = [w for w, d in df.items()
             if d >= MIN_DF and low[w.lower()] <= 0.1 * cap[w]
             and w.lower() not in wordlist and w not in cavelib.MONTHS_DAYS]
    return sorted(names), df


def candidates():
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)
    shards = sorted(rng.sample(range(41), N_SHARDS))
    n = 0
    with open(OUT / "candidates.jsonl", "w") as f:
        for s in shards:
            path = hf_hub_download("wikimedia/wikipedia", f"20231101.en/train-{s:05d}-of-00041.parquet",
                                   repo_type="dataset")
            rows = pq.read_table(path, columns=["id", "title", "text"]).to_pylist()
            for r in rows:
                txt = clean(r["text"])
                if keep_article(r["title"], txt):
                    f.write(json.dumps({"id": r["id"], "title": r["title"], "text": txt}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"shard {s}: {len(rows):,} rows -> {n:,} candidate articles so far", flush=True)


def split():
    rng = random.Random(SEED)
    topic = {d["id"]: d["topic"] for d in map(json.loads, open(OUT / "topics.jsonl"))}
    arts = [a for a in map(json.loads, open(OUT / "candidates.jsonl")) if topic.get(a["id"], "other") != "other"]
    for a in arts:
        a["topic"] = topic[a["id"]]
    arts.sort(key=lambda a: a["id"])
    rng.shuffle(arts)
    print(f"on-topic articles: {len(arts):,}, {sum(len(a['text'].encode()) for a in arts)/1e6:.1f} MB | "
          f"{dict(collections.Counter(a['topic'] for a in arts).most_common())}")

    held, train, hb, tb = [], [], 0, 0
    for a in arts:
        b = len(a["text"].encode())
        if hb < HELD_SRC_BYTES:
            held.append(a); hb += b
        elif tb < TRAIN_SRC_BYTES:
            train.append(a); tb += b
        else:
            break
    print(f"train {len(train)} articles {tb/1e6:.1f} MB | held-out {len(held)} articles {hb/1e6:.2f} MB")
    if tb < TRAIN_SRC_BYTES:
        print("WARNING: not enough on-topic text for the train target -> classify more articles or add a shard")

    W = cavelib.load_wordlist()
    names, df = core_names(train, W)
    (Path(__file__).parent / "core_names.txt").write_text("\n".join(names) + "\n")
    print(f"core names (df >= {MIN_DF}): {len(names)} | {sorted(names, key=lambda w: -df[w])}")

    with open(OUT / "articles.jsonl", "w") as fa, open(OUT / "chunks.jsonl", "w") as fc:
        for sp, group in [("train", train), ("held", held)]:
            for a in group:
                subj = cavelib.subject_names(a["title"], a["text"], W)
                fa.write(json.dumps({**a, "split": sp, "subject": subj}, ensure_ascii=False) + "\n")
                for k, c in enumerate(chunks(a["text"])):
                    fc.write(json.dumps({"chunk_id": f"{a['id']}_{k}", "article_id": a["id"], "split": sp,
                                         "topic": a["topic"], "title": a["title"], "subject": subj, "text": c},
                                        ensure_ascii=False) + "\n")
    print("n chunks:", sum(1 for _ in open(OUT / "chunks.jsonl")))


if __name__ == "__main__":
    {"candidates": candidates, "split": split}[sys.argv[1]]()
