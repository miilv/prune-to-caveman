"""Step 3b: rewrite source chunks into caveman speech with Haiku 5.5, check every output against the rules,
and send one repair request when too many words break them.

  python generate.py --pilot 20     # 20 seeded train chunks -> data/pilot_<model>.jsonl
  python generate.py                # all chunks             -> data/caveman_raw.jsonl   (resumable)
"""
import argparse, asyncio, collections, json, os, random, sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import cavelib, gateway

SEED = 0
MAX_OOV = 0.03            # accept an output when <= 3% of its words break rules 1/5
CONCURRENCY = int(os.environ.get("CONCURRENCY", 16))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 16000))   # a few chunks think past 16k; their retry pass uses 48000
MODEL = gateway.MODEL
DATA = Path(__file__).parent / "data"
REPAIR = ("These words break the rules: {bad}.\nRewrite the WHOLE caveman text again. Replace each of these words "
          "with list words (or allowed names). Keep everything else the same. Output only the caveman text.")


async def rewrite(client, sem, system, chunk, W, names, out, stats):
    allowed = names | set(chunk["subject"])
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": cavelib.user_message(chunk["title"], chunk["subject"], chunk["text"])}]
    usage_all, rec = collections.Counter(), None
    async with sem:
        for attempt in range(2):
            try:
                text, usage = await gateway.chat(client, msgs, max_tokens=MAX_TOKENS, model=MODEL)
            except Exception as e:
                print(f"{chunk['chunk_id']}: failed ({e})", flush=True)
                return
            usage_all.update({"in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0),
                              "think": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                              "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)})
            text = cavelib.normalize(text, W, allowed)
            chk = cavelib.check(text, W, allowed)
            rec = {**{k: chunk[k] for k in ("chunk_id", "article_id", "split", "topic", "title", "subject")},
                   "caveman": text, "attempts": attempt + 1, "oov_rate": chk["oov_rate"], "bad": chk["bad"],
                   "bad_punct": chk["bad_punct"], "len_ratio": len(text) / len(chunk["text"]),
                   "usage": dict(usage_all), "first_oov": rec["first_oov"] if rec else chk["oov_rate"]}
            if chk["oov_rate"] <= MAX_OOV:
                break
            bad = ", ".join(sorted(set(chk["bad"]))[:80])
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": REPAIR.format(bad=bad)}]
    rec["accepted"] = rec["oov_rate"] <= MAX_OOV
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    stats["n"] += 1; stats["ok"] += rec["accepted"]; stats.update(usage_all)
    if stats["n"] % 10 == 0 or stats["n"] == stats["total"]:
        print(f"{stats['n']}/{stats['total']} done, {stats['ok']} accepted | tokens in {stats['in']:,} "
              f"(cached {stats['cached']:,}) out {stats['out']:,} (thinking {stats['think']:,})", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--model", default=gateway.MODEL)
    args = ap.parse_args()
    global MODEL
    MODEL = args.model
    W, names = cavelib.load_wordlist(), cavelib.load_names()
    system = cavelib.system_prompt(W, names)
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
    if args.pilot:
        chunks = random.Random(SEED).sample([c for c in chunks if c["split"] == "train"], args.pilot)
        path = DATA / f"pilot_{MODEL.split('/')[-1]}.jsonl"
        path.unlink(missing_ok=True)
    else:
        path = DATA / "caveman_raw.jsonl"
        done = {json.loads(l)["chunk_id"] for l in open(path)} if path.exists() else set()
        chunks = [c for c in chunks if c["chunk_id"] not in done]
    print(f"system prompt ~{len(system) // 4:,} tokens | {len(chunks):,} chunks to rewrite", flush=True)
    stats = collections.Counter(total=len(chunks))
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        with open(path, "a") as out:
            await asyncio.gather(*(rewrite(client, sem, system, c, W, names, out, stats) for c in chunks))


if __name__ == "__main__":
    asyncio.run(main())
