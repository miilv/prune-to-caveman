# Corpus pipeline: Wikipedia → caveman English

How `corpus/` was built. The notebook does **not** need any of this: it downloads the frozen corpus. These scripts call
an LLM, so re-running them costs money and will not reproduce the corpus byte for byte (sampling on the API side). Every
other step is seeded (`SEED = 0`).

| step | script | output |
|---|---|---|
| 1. sources | `prepare_sources.py candidates` | 2 seeded shards of `wikimedia/wikipedia` `20231101.en` → cleaned articles (reference sections cut, list-like lines dropped, 3–40k chars, < 3 % non-ASCII) |
| 2. topics | `classify_topics.py` | `claude-haiku-5-5` labels 16k seeded articles with one of 8 topics, 50 per request; `other` (modern sport, films, companies, towns …) is dropped |
| 3. split | `prepare_sources.py split` | **held-out first** (1.2 MB of English source), then train (14 MB), split **by article**; `core_names.txt` = Capitalized words in ≥ 50 train articles that are (almost) never lowercase; chunks of ≤ 6,000 chars along paragraph boundaries |
| 4. rewrite | `generate.py` | each chunk rewritten by `claude-haiku-5-5` with `style_guide.md` + 4 worked examples (`examples.json`); every output is checked word by word (`cavelib.check`); if > 3 % of the words break the rules, one repair turn lists the offending words |
| 5. freeze | `finalize.py` | punctuation normalised; one corpus-wide replacement dictionary for leftover off-list words (built by Haiku from the word *types*, each entry re-checked); sentences that still break the rules are dropped; chunks joined back into articles → `corpus/` |

## The language

* **Words:** `wordlist.txt` = Basic English 850 (Ogden; list from Wiktionary *Appendix:Basic English word list*) ∪
  `caveman_additions.txt` (111 words: *rock, cave, mammoth, spear, tribe, kill, hunt, one … ten, they, we* …) minus *a*, *the* → 961 words.
  Lowercase only, base forms only (no *-s, -ed, -ing, -ly, -er*).
* **Names:** `core_names.txt` (Rome, Egypt, Napoleon, …) plus the names in the title of the article being rewritten
  (*Battle of Hastings* → *Hastings*), always Capitalized. Every other name is described ("Seine" → "big river").
* **Other symbols:** digits and `. , ? !`.
* **Voice:** neutral, no "grug", no jokes; facts and order of the original are kept.

## Running it

```bash
uv venv && uv pip install httpx pyarrow huggingface_hub
export LLM_BASE_URL=https://.../v1 LLM_API_KEY=...   # any OpenAI-compatible endpoint serving claude-haiku-5-5
python prepare_sources.py candidates
python classify_topics.py
python prepare_sources.py split
python generate.py --pilot 20               # look at data/pilot_*.jsonl first
CONCURRENCY=128 python generate.py          # resumable
python finalize.py
```

Note on cost: `claude-haiku-5-5` thinks a lot on this task (~10k reasoning tokens per 6k-char chunk; with thinking
disabled 15–30 % of the words break the rules and repair turns do not converge). The final run used about 45 M output
tokens; prompt caching of the 2.6k-token system prompt covered ~78 % of the input tokens.

`replacements.json` is the corpus-wide dictionary that `finalize.py` built (it caches it in `data/replacements.json`).
Intermediate files go to `corpus_pipeline/data/` (not in the repository).
