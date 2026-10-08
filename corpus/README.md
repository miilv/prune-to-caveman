# Caveman English corpus

| file | content |
|---|---|
| `caveman_train.json` | list of train articles (str), caveman English — token usage for pruning is counted here only |
| `caveman_heldout.json` | list of held-out articles (str), caveman English — different articles from train |
| `heldout_source_en.json` | the original English Wikipedia text of the held-out articles, same order (used only as an out-of-language probe) |
| `articles.jsonl` | one line per article: Wikipedia page id, title, URL, split, topic, whether every chunk survived |
| `stats.json` | sizes, rule-violation rates, dropped sentences, word types |

**Source and licence.** English Wikipedia, dump `20231101.en` via the Hugging Face dataset `wikimedia/wikipedia`
(two shards picked with `random.Random(0)`). Wikipedia text is licensed **CC BY-SA 4.0**; the caveman versions are an
adaptation (rewritten by `claude-haiku-5-5`) and are distributed under the same licence. The authors of every article
are listed in its history at the URL given in `articles.jsonl`.

**Language.** Lowercase words from a closed list of 961 words (Basic English 850 + 111 caveman words − *a*, *the*),
~140 Capitalized core names (+ the names in the article title), digits and `. , ? !`. How it was built:
[`../corpus_pipeline/`](../corpus_pipeline/).
