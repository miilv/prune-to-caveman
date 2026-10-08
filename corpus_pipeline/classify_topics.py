"""Step 3a': label a seeded subset of candidate articles with one topic each (Haiku 5.5, 50 articles/request).
Input: data/candidates.jsonl. Output: data/topics.jsonl ({"id", "topic"} per article; resumable)."""
import asyncio, json, random, re, sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import gateway

SEED = 0
N_CLASSIFY = 16000
BATCH = 50
CONCURRENCY = 16
DATA = Path(__file__).parent / "data"
TOPICS = {
    "history": "ancient and medieval history: empires, wars, battles, kings, dynasties, ancient cities and peoples",
    "people_pre1900": "people who lived mainly before 1900: rulers, explorers, inventors, thinkers, artists",
    "nature": "animals, plants, fungi, weather, ecology",
    "earth_places": "natural geography: mountains, rivers, lakes, islands, volcanoes, deserts, seas (NOT towns or districts)",
    "science": "astronomy, physics, chemistry, the human body, medicine, mathematics",
    "myth_religion": "gods, myths, legends, religions, rituals",
    "food_tools_crafts": "food, cooking, farming, weapons, tools, building, crafts, old inventions",
    "other": "everything else: modern sport and athletes, music, films, TV, games, companies, modern politics, "
             "towns, villages, administrative units, schools, organizations, roads, stations, modern people",
}
PROMPT = ("Label each Wikipedia article with exactly ONE topic id.\n\nTOPICS\n"
          + "\n".join(f"{k}: {v}" for k, v in TOPICS.items())
          + "\n\nIf unsure, or if the article is about something modern (after about 1900), use other.\n"
            "Answer with one line per article, in the same order, formatted exactly as: <number> <topic id>\n"
            "No other text.")


def parse(answer, n):
    labels = {}
    for line in answer.splitlines():
        m = re.match(r"\s*(\d+)[\s.:)-]+([a-z_0-9]+)", line)
        if m and 1 <= int(m.group(1)) <= n and m.group(2) in TOPICS:
            labels[int(m.group(1))] = m.group(2)
    return labels


async def run_batch(client, sem, batch, out, stats):
    lines = [f"{i + 1}. {a['title']} | {a['text'][:400].replace(chr(10), ' ')}" for i, a in enumerate(batch)]
    async with sem:
        try:
            text, usage = await gateway.chat(client, [{"role": "system", "content": PROMPT},
                                                      {"role": "user", "content": "\n".join(lines)}], max_tokens=8000)
        except Exception as e:
            print("batch failed:", e, flush=True); return
    labels = parse(text, len(batch))
    for i, a in enumerate(batch):
        if i + 1 in labels:
            out.write(json.dumps({"id": a["id"], "topic": labels[i + 1]}) + "\n")
    out.flush()
    stats["done"] += len(labels); stats["missing"] += len(batch) - len(labels)
    for k in ("prompt_tokens", "completion_tokens"):
        stats[k] += usage.get(k, 0)
    print(f"labelled {stats['done']:,} (missing {stats['missing']}) | tokens in {stats['prompt_tokens']:,} "
          f"out {stats['completion_tokens']:,}", flush=True)


async def main():
    cands = [json.loads(l) for l in open(DATA / "candidates.jsonl")]
    random.Random(SEED).shuffle(cands)
    cands = cands[:N_CLASSIFY]
    done = set()
    if (DATA / "topics.jsonl").exists():
        done = {json.loads(l)["id"] for l in open(DATA / "topics.jsonl")}
    todo = [a for a in cands if a["id"] not in done]
    print(f"{len(done):,} already labelled, {len(todo):,} to go")
    stats = {"done": len(done), "missing": 0, "prompt_tokens": 0, "completion_tokens": 0}
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        with open(DATA / "topics.jsonl", "a") as out:
            await asyncio.gather(*(run_batch(client, sem, todo[i:i + BATCH], out, stats)
                                   for i in range(0, len(todo), BATCH)))


if __name__ == "__main__":
    asyncio.run(main())
