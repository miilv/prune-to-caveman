"""Summary of the generation run for the report -> ../results/generation_stats.json"""
import json, re, statistics
from pathlib import Path

HERE = Path(__file__).parent
raw = [json.loads(l) for l in open(HERE / "data" / "caveman_raw.jsonl")]
n_chunks = sum(1 for _ in open(HERE / "data" / "chunks.jsonl"))
train = json.load(open(HERE.parent / "corpus" / "caveman_train.json"))
u = {k: sum(r["usage"].get(k, 0) for r in raw) for k in ("in", "out", "think", "cached")}
s = {"chunks": n_chunks, "chunks_done": len(raw), "accepted": sum(r["accepted"] for r in raw) / len(raw),
     "repaired": sum(r["attempts"] > 1 for r in raw) / len(raw),
     "first_pass_offlist": statistics.mean(r["first_oov"] for r in raw),
     "final_offlist": statistics.mean(r["oov_rate"] for r in raw),
     "len_ratio": statistics.mean(r["len_ratio"] for r in raw),
     "train_words": sum(len(d.split()) for d in train),
     "in_tokens_m": u["in"] / 1e6, "out_tokens_m": u["out"] / 1e6, "think_share": u["think"] / max(u["out"], 1),
     "cached_share": u["cached"] / max(u["in"], 1)}
(HERE.parent / "results").mkdir(exist_ok=True)
(HERE.parent / "results" / "generation_stats.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s, indent=1))
