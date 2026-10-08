"""Fill report.typ from ../results/results.json and ../corpus/stats.json, then: typst compile report.typ"""
import json
from pathlib import Path

HERE = Path(__file__).parent
R = json.load(open(HERE.parent / "results" / "results.json"))
S = json.load(open(HERE.parent / "corpus" / "stats.json"))
G = json.load(open(HERE.parent / "results" / "generation_stats.json"))

f2, f3, f4 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:.3f}"), (lambda x: f"{x:.4f}")
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
k = lambda x: f"{x:,}"
p1 = {r["tokenizer"]: r for r in R["part1"]}
q = p1["Qwen2.5 (151k)"]
p2, p3 = R["part2"], R["part3"]
r0, r1, r1r = p3["bpb"]; e0, e1 = p3["bpb_en"]
P0, P1 = p3["params"]; C0, C1 = p3["ckpt_mb"]
sw = {r["run"]: r for r in R["part4"]}
m1 = sw["m=1"]
env, times = R["env"], R["times"]
total_s = sum(times.values())

def row(*cells):
    return ", ".join(f"[{c}]" for c in cells) + ","

t1 = "\n".join(row(name, k(r["vocab"]), f2(r["bytes/token"]), f3(r["fertility"]), k(r["distinct ids"]),
                   pct(r["share of vocab"], 2), k(r["ids for 99% of occurrences"])) for name, r in p1.items())
t3 = "\n".join(row(name, k(r["vocab"]), f"{r['params (M)']:.1f}", pct(r["held identical"]), f4(r["length ratio"]),
                   f4(r["tokens/byte"]), f4(r["bits/byte"]), f3(r["English length ratio"]), f3(r["English bits/byte"]))
               for name, r in sw.items())
b1 = "\n".join(row(r["new tokens"], f4(r["fertility (train)"]), f4(r["tokens/byte (held)"]), f4(r["bits/byte (held)"]),
                   f4(r["bits/byte after emb. fine-tune"]) if r["bits/byte after emb. fine-tune"] else "–") for r in R["B1"])
b1_ex = R["B1"][-1]["examples"].replace("␣", "␣")
b4 = "\n".join(row(r["model"], r["batch"], f"{r['tokens/s']:.0f}", f"{r['ms/step']:.2f}", f"{r['peak MiB']:.0f}") for r in R["B4"])
B4 = {(r["model"], r["batch"]): r for r in R["B4"]}
b4o, b4p = B4[("original", 1)], B4[("pruned m=1", 1)]
b3 = R["B3"]

typ = f'''#set page(paper: "a4", margin: (x: 1.4cm, y: 1.25cm), numbering: "1")
#set text(size: 9pt, font: "New Computer Modern")
#set par(justify: true, leading: 0.52em, spacing: 0.75em)
#set heading(numbering: "1.")
#show heading: set text(size: 10pt)
#show heading: set block(above: 0.9em, below: 0.5em)
#set table(inset: (x: 3.5pt, y: 2pt), stroke: (x, y) => if y == 0 {{ (bottom: 0.6pt) }})
#show table: set text(size: 8pt)
#show figure.caption: set text(size: 8pt)
#show raw: set text(size: 8.4pt)

#align(center)[#text(size: 13pt, weight: "bold")[Homework 1 — Tokenizer pruning for caveman English] \\
Ilia Mikhalchuk · Modern Methods and Algorithms of Generative AI · Skoltech, Fall 2026 · #link("https://github.com/miilv/prune-to-caveman")[github.com/miilv/prune-to-caveman]]

= Language, corpus and setup
*Language.* _Caveman English_, a closed-vocabulary register of English: lowercase base forms from a list of 961 words
(Basic English 850 + 111 caveman words such as _mammoth, spear, tribe, kill, hunt_, minus _a_ and _the_), no inflections,
about 140 frequent proper names (plus the names in the article's own title) kept Capitalized, digits and `. , ? !`.
Example: _"Elephantomyia be dead kind of long leg fly, from long ago. people know it only from clear tree stone."_ (amber).
A synthetic register rather than a natural language, chosen as the extreme case of the homework's setting.

*Corpus.* English Wikipedia (`wikimedia/wikipedia` `20231101.en`, two shards drawn with seed 0, *CC BY-SA 4.0*);
articles on history, nature, places, science, myth and crafts (topic labelled by an LLM, modern topics dropped), split
*by article* into train and held-out before generation, then rewritten paragraph by paragraph by `claude-haiku-5-5`.
Every output was checked word by word; {pct(G["first_pass_offlist"])} of words broke the rules
in the first pass, {pct(S["raw_offlist_rate"], 2)} after one repair turn; the rest was fixed by a corpus-wide replacement dictionary and
{pct(S["sentences_dropped"] / (S["sentences_kept"] + S["sentences_dropped"]), 2)} of sentences were dropped.
*Train:* {k(S["train_docs"])} articles, {S["train_bytes"] / 1e6:.2f} MB, {k(G["train_words"])} words.
*Held-out:* {k(S["held_docs"])} other articles, {S["held_bytes"] / 1e3:.0f} KB (all of it is evaluated). The English originals of the held-out
articles ({S["held_source_bytes"] / 1e3:.0f} KB) are used only as an _out-of-language probe_ (first 4,000 chars of 60 documents).

*Setup.* `Qwen/Qwen2.5-0.5B` (byte-level BPE, 151,643 + 22 added tokens, tied embeddings), seed 0, one {env["device"]},
{env["dtype"].replace("torch.", "")} weights with float32 cross-entropy, torch {env["torch"]}, transformers {env["transformers"]},
tokenizers {env["tokenizers"]}. The whole notebook runs in {total_s / 60:.1f} min on this GPU (on a T4 it switches to fp16,
which gives the same bits per byte to 4 decimals). Bits per byte follows §5 of the assignment exactly (`<|endoftext|>`
prefix, non-overlapping windows of 512 tokens); my batched implementation is checked against the starter's function in the notebook.

= Token usage (Part 1)
#figure(table(columns: 7, align: (left, right, right, right, right, right, right),
[tokenizer], [vocab], [bytes/token], [fertility], [distinct ids], [share], [ids for 99%],
{t1}
), caption: [Train corpus ({S["train_bytes"] / 1e6:.1f} MB). Fertility = tokens per whitespace word.]) <t1>

Qwen2.5 needs only {k(q["distinct ids"])} of its 151,643 ids ({pct(q["share of vocab"], 2)}) for the whole train corpus, and
{k(q["ids for 99% of occurrences"])} ids cover 99% of all token occurrences (@cov). On the held-out side the same tokenizer uses
{k(R["held_distinct"]["caveman"])} ids for the caveman text and {k(R["held_distinct"]["english"])} for the English originals of the *same* articles.
All four tokenizers are practically equivalent here (fertility {f2(min(r["fertility"] for r in p1.values()))}–{f2(max(r["fertility"] for r in p1.values()))}): every list word is one
token with its leading space, the rest are punctuation, digits and pieces of rare names. Qwen is slightly worse only because
its pre-tokenizer splits numbers into single digits and caveman text is full of years ("year 1453"): with the numbers removed Qwen and cl100k both give 1.172. A bigger vocabulary buys nothing for this language.

#figure(image("../figures/coverage.png", width: 80%), caption: [Coverage of token occurrences by the $k$ most frequent ids.
Left: train corpus, four tokenizers. Right: Qwen2.5 on the held-out articles, caveman vs their English originals.]) <cov>

= Pruning the tokenizer (Part 2)
With $m = 1$ the vocabulary shrinks from {k(p2["vocab"][0])} to *{k(p2["vocab"][1])}* tokens and the merges from {k(p2["merges"][0])} to
*{k(p2["merges"][1])}* (22 added tokens kept, renumbered after the vocabulary; {k(R["closure"]["added"])} tokens are added only by the closure).
Checks: (1) `decode(encode(x)) == x` holds for all held-out documents and also for the English originals (the 256 byte-level
characters are always kept); (2) all {k(S["train_docs"])} train articles are tokenized identically; (3) on held-out text
{pct(p2["held"]["identical_fraction"])} of the articles ({pct(p2["held_paragraphs"]["identical_fraction"])} of the paragraphs) are identical, length ratio
{f4(p2["held"]["length_ratio"])} — the differences are names that never occurred in train; the English originals: length ratio {f3(p2["english"]["length_ratio"])}.

*Why (2) must hold.* Pre-tokenization does not depend on the vocabulary. Inside a pre-token BPE always applies the
applicable merge of lowest rank. Every token produced while tokenizing the train corpus is either final (count ≥ 1, so kept)
or an intermediate that is later merged into a final token; since Qwen's merges have unique results, these intermediates
are exactly the nodes of the final token's merge tree, which the closure keeps. So every merge that ever fired survives
in the same order, and the removed merges never fired: at every step the pruned tokenizer makes the same choice.

*Closure example.* `␣this` (merge \\#163) is built as `␣th` + `is`, and `␣th` = `␣t` + `h`. Neither `␣th` nor `␣t` ever
occurs as a final token, so a frequency-only vocabulary drops them. Then `tokenizers` refuses to load the file (_"Token `Ġt` out of
vocabulary"_); if we also drop the merges with missing inputs, `␣this` can no longer be built and becomes `␣` + `this`.
In total {k(R["no_closure"]["unreachable"])} kept tokens become unreachable dead rows, no train article is tokenized identically
any more and sequences get {f2(R["no_closure"]["length_ratio"])}× longer.

= Pruning the model (Part 3)
#figure(table(columns: 7, align: (left,) + (right,) * 6,
[], [params], [bf16 checkpoint], [bits/byte], [tokens/byte], [bits/byte EN probe], [tokens/byte EN probe],
[original], [{P0 / 1e6:.1f} M], [{C0:.0f} MB], [{f4(r0["bits_per_byte"])}], [{f4(r0["tokens_per_byte"])}], [{f3(e0["bits_per_byte"])}], [{f4(e0["tokens_per_byte"])}],
[pruned $m = 1$], [{P1 / 1e6:.1f} M], [{C1:.0f} MB], [{f4(r1["bits_per_byte"])}], [{f4(r1["tokens_per_byte"])}], [{f3(e1["bits_per_byte"])}], [{f4(e1["tokens_per_byte"])}],
[reloaded from disk], [{P1 / 1e6:.1f} M], [{C1:.0f} MB], [{f4(r1r["bits_per_byte"])}], [{f4(r1r["tokens_per_byte"])}], [], [],
), caption: [Held-out caveman ({k(r0["tokens"])} tokens, {r0["bytes"] / 1e3:.0f} KB) and the English probe. Embedding rows: 151,936 → {k(m1["vocab"] + 22 + (-(m1["vocab"] + 22)) % 64)} (padded to 64).]) <t2>

Embeddings are sliced by index (no `resize_token_embeddings`), the tied head shares the new matrix, `vocab_size` and the
special-token ids in `config` / `generation_config` are remapped (otherwise `generate` stops on a wrong id).
Parameters drop by {pct(1 - P1 / P0)} and the checkpoint by {pct(1 - C1 / C0)}. Greedy generation (float32) of the pruned model is identical
to the original's as long as the original's choice is still in the vocabulary (1 of 3 prompts; the other two diverge at `␣pact` and
`aming`, which were pruned); the reloaded checkpoint generates the same text and reproduces bits per byte exactly.

*Why bits per byte improves ({f4(r0["bits_per_byte"])} → {f4(r1["bits_per_byte"])}) and why the model is not better.* The pruned head is a
softmax over {k(m1["vocab"] + 22)} logits instead of 151k; all kept logits are unchanged, so for every target token that is still in the
vocabulary $p_"pruned"(t) = p(t) / sum_(t' "kept") p(t') >= p(t)$. The mass the model put on {k(151_643 - m1["vocab"])} removed tokens (other scripts, code,
rare English) is redistributed over the kept ones, and bits per byte, which scores the exact token sequence, drops by the log of that factor.
Nothing in the network changed — the same number comes from masking the logits — we only injected a prior from train-corpus
statistics about which tokens can occur. The price shows on text outside the domain: on ordinary English the pruned model
needs {f2(e1["tokens_per_byte"] / e0["tokens_per_byte"])}× more tokens and bits per byte rises from {f3(e0["bits_per_byte"])} to {f3(e1["bits_per_byte"])}.

= How far can we go? (Part 4)
#figure(image("../figures/sweep.png", width: 88%), caption: [Threshold sweep. Black markers: random control with the same vocabulary size (closure respected).]) <sweep>
#figure(table(columns: 9, align: (left,) + (right,) * 8,
[run], [vocab], [params, M], [held identical], [length ratio], [tokens/byte], [bits/byte], [EN length ratio], [EN bits/byte],
{t3}
), caption: [Threshold sweep and random control on the held-out caveman text and the English probe (EN).]) <t3>

*Random control.* Keeping the same number of tokens at random (with closure) instead of the frequent ones makes the caveman
text {f2(sw["random, size of m=1"]["length ratio"])}× longer and bits per byte jumps to {f3(sw["random, size of m=1"]["bits/byte"])} (vs {f4(m1["bits/byte"])}). The gain is
therefore not "fewer tokens are fine": it comes from the extremely skewed usage distribution, which the frequency threshold
follows.

*Knee and what to ship.* Bits per byte stays below the original up to $m = 100$ (vocabulary ≈ 2k: roughly the 961 words in
their 1–2 spellings plus punctuation and digits) and breaks at $m = 500$, where list words start to split. In *parameters*, however,
the knee is at $m = 1$ already: the embedding share falls from 27.6% to 1.6%, and every further step saves at most 1.3% of the
model while the held-out agreement drops from {pct(m1["held identical"])} to {pct(sw["m=20"]["held identical"])} ($m = 20$). I would ship *$m = 1$*: −{pct(1 - P1 / P0)}
parameters, the best bits per byte, exactly the original tokenization on everything seen in training, and the most
robust handling of unseen names.

= Bonuses
*B3 — tokenizer trained from scratch.* A byte-level BPE trained on the caveman corpus with Qwen's pre-tokenizer and the same vocabulary size
gives almost the same tokens/byte, but plugged into Qwen with random embeddings the model is useless:
{f3(b3["random"]["bits/byte"])} bits/byte — worse than a uniform distribution — and generation degenerates into repeating one token.
Initialising each new token as the mean of the Qwen embeddings of its pieces (FVT) restores {f3(b3["mean of Qwen pieces"]["bits/byte"])} bits/byte without any training.
*B1 — vocabulary extension.* New merges trained with `tokenizers` on the caveman corpus, appended after the pruned vocabulary
(ranked by how often their inputs are adjacent), mean-initialised: fertility {f4(R["B1"][0]["fertility (train)"])} → {f4(R["B1"][-1]["fertility (train)"])}
with {R["B1"][-1]["new tokens"]} new tokens — little room is left, since every list word is already one token and only rare names split.
An embedding-only fine-tune (100 steps) lowers bits per byte to ≈{f3(R["B1"][0]["bits/byte after emb. fine-tune"])} for the pruned and the extended model alike.
*B4 — serving.* Peak memory {b4o["peak MiB"]:.0f} → {b4p["peak MiB"]:.0f} MiB at batch 1; decode throughput changes by only
{100 * (b4p["tokens/s"] / b4o["tokens/s"] - 1):+.0f}% (batch 1): GPU decoding of a 0.5B model is dominated by the 24 layers and kernel launches, not by the head.

= Limitations
The corpus is LLM-generated: its style is Haiku's, it is shorter than the source (caveman/English ≈ {G["len_ratio"]:.2f}) and some
paragraphs are condensed; held-out text comes from the same generator, so it is in-distribution by construction. Caveman English
is a register of English, not a natural language, and its closed vocabulary makes pruning much cleaner than for a real language.
Re-generating the corpus would not reproduce it byte for byte (hence the frozen files). One model, one seed, one corpus. The original model does not "speak" caveman (it continues prompts in normal English);
pruning restricts the vocabulary but not the grammar.

*AI assistance.* The notebook, the corpus pipeline and this report were written with an AI coding assistant (Claude Code with
Claude Opus 5.5) under my direction; the corpus itself was generated by `claude-haiku-5-5` (≈{G["out_tokens_m"]:.0f} M output tokens).

#pagebreak()
= Appendix
#figure(table(columns: 5, align: (right,) * 5,
[new tokens], [fertility (train)], [tokens/byte (held)], [bits/byte (held)], [after embedding fine-tune],
{b1}
), caption: [B1 — vocabulary extension on top of the $m = 1$ tokenizer. First new tokens: {b1_ex}.])

#figure(table(columns: 5, align: (left, right, right, right, right),
[model], [batch], [tokens/s], [ms/step], [peak MiB],
{b4}
), caption: [B4 — greedy decoding of 128 new tokens, {env["device"]}, {env["dtype"].replace("torch.", "")}.])

*Corpus generation.* {k(G["chunks"])} chunks of ≤ 6,000 characters; {pct(G["accepted"])} accepted after at most one repair turn ({G["chunks"] - G["chunks_done"]} chunks never produced an answer and are missing); ≈{G["think_share"]:.0%} of
the {G["out_tokens_m"]:.0f} M output tokens were the model's thinking; prompt caching served {pct(G["cached_share"], 0)} of the {G["in_tokens_m"]:.0f} M input tokens.
With thinking disabled 15–30% of the words broke the rules and repair turns did not converge, so thinking was kept on.
The full pipeline (`corpus_pipeline/`) and the style guide given to the model are in the repository.

*Reproduce.* Open `hw1_mikhalchuk.ipynb` in Colab (GPU) or any machine with a GPU and run all cells; it downloads the
frozen corpus from the repository and Qwen from the Hub. Runtime per part (s, {env["device"]}): {", ".join(f"{a} {b:.0f}" for a, b in times.items())}.
'''
(HERE / "report.typ").write_text(typ)
print("report.typ written")

# ---------------------------------------------------------------- README
sweep_md = "\n".join(f"| {name} | {r['vocab']:,} | {r['params (M)']:.1f} M | {pct(r['held identical'])} | {f4(r['length ratio'])} | "
                     f"{f4(r['bits/byte'])} | {f3(r['English bits/byte'])} |" for name, r in sw.items())
readme = f'''# prune-to-caveman

**Tokenizer and embedding pruning of `Qwen/Qwen2.5-0.5B` for _caveman English_** — Homework 1 of *Modern Methods and
Algorithms of Generative AI* (Skoltech, Fall 2026), Ilia Mikhalchuk.

> *mammoth be big animal with long hair. people hunt it with spear long ago.*

The model's 151,643-token vocabulary is cut to **{k(m1["vocab"])} tokens** (−{pct(1 - P1 / P0)} parameters, checkpoint
{C0:.0f} → {C1:.0f} MB) without changing a single token of the train corpus' tokenization, and held-out bits per byte goes
{f4(r0["bits_per_byte"])} → {f4(r1["bits_per_byte"])}.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miilv/prune-to-caveman/blob/main/hw1_mikhalchuk.ipynb)
· **Report:** [`report/hw1_mikhalchuk_report.pdf`](report/hw1_mikhalchuk_report.pdf)
· **Pruned model:** [GitHub release `v1.0`](https://github.com/miilv/prune-to-caveman/releases/tag/v1.0) (`tokenizer.json`, `config.json`, `model.safetensors`)

## The language

*Caveman English* is a closed-vocabulary register of English: lowercase base forms from **961 words** (Basic English 850 +
111 caveman words such as *mammoth, spear, tribe, hunt*, minus *a* / *the*), no inflections, ~140 frequent proper names
kept Capitalized (plus the names in the article's own title), digits and `. , ? !`. The corpus is English Wikipedia
(history, nature, places, science, myth, crafts; CC BY-SA 4.0) rewritten by `claude-haiku-5-5` and checked word by word.

| English Wikipedia | caveman English (this corpus) |
|---|---|
| *Elephantomyia brevipalpa is an extinct species of crane fly in the family Limoniidae. The species is solely known from the Middle Eocene Baltic amber deposits in the Baltic Sea region of Europe.* | *this one Elephantomyia be dead kind of long leg fly, from long ago. people know it only from clear tree stone. this stone come from land near sea, in Europe.* |
| *Amphiprion latezonatus, also known as the wide-band anemonefish, is a species of anemonefish found in subtropical waters off the east coast of Australia.* | *Amphiprion, also name wide band fish, be kind of sea flower fish. it live in sea water off east edge of Australia.* |

Train: **{k(S["train_docs"])} articles, {S["train_bytes"] / 1e6:.1f} MB**; held-out: **{k(S["held_docs"])} other articles, {S["held_bytes"] / 1e3:.0f} KB**
(split by article before generation), plus the English originals of the held-out articles as an out-of-language probe.

## Results

| | original | pruned, m = 1 |
|---|---|---|
| vocabulary / merges | 151,643 / 151,387 | {k(p2["vocab"][1])} / {k(p2["merges"][1])} |
| parameters | {P0 / 1e6:.1f} M | {P1 / 1e6:.1f} M (−{pct(1 - P1 / P0)}) |
| bf16 checkpoint | {C0:.0f} MB | {C1:.0f} MB |
| train corpus tokenized identically | – | {pct(p2["train"]["identical_fraction"], 0)} of articles |
| held-out bits per byte | {f4(r0["bits_per_byte"])} | {f4(r1["bits_per_byte"])} |
| English probe bits per byte | {f3(e0["bits_per_byte"])} | {f3(e1["bits_per_byte"])} |

Qwen2.5 uses **{k(q["distinct ids"])} of 151,643 ids ({pct(q["share of vocab"], 2)})** on the train corpus; {k(q["ids for 99% of occurrences"])} ids cover 99% of all occurrences.

| run | vocab | params | held-out identical | length ratio | bits/byte | English bits/byte |
|---|---|---|---|---|---|---|
{sweep_md}

![coverage](figures/coverage.png)
![sweep](figures/sweep.png)

* **Why bits per byte improves:** the pruned softmax renormalises over the kept tokens, so every kept target gets
  $p/\\sum_{{kept}} p \\ge p$ — a prior from corpus statistics, not a better model; on ordinary English it gets much worse.
* **Random control:** the same number of random tokens (with merge closure) makes caveman text {f2(sw["random, size of m=1"]["length ratio"])}× longer and
  {f3(sw["random, size of m=1"]["bits/byte"])} bits/byte — the gain comes from the skewed usage distribution.
* **What to ship:** m = 1. After the first cut the embedding is 1.6% of the model; larger thresholds save at most 1.3% more and lose robustness.
* **Bonuses:** B1 vocabulary extension (little to gain: every list word is already one token), B3 tokenizer trained from
  scratch (random embeddings → {f3(b3["random"]["bits/byte"])} bits/byte; mean-of-pieces init → {f3(b3["mean of Qwen pieces"]["bits/byte"])}), B4 serving (peak memory {b4o["peak MiB"]:.0f} → {b4p["peak MiB"]:.0f} MiB).

## Repository

```
hw1_mikhalchuk.ipynb        the homework notebook, executed (Parts 1–4, bonuses B1, B3, B4)
report/                     2-page report (PDF) + its Typst source and generator
corpus/                     frozen caveman corpus: train / held-out / English originals of held-out + metadata
corpus_pipeline/            how the corpus was built (Wikipedia → topic filter → split → Haiku rewrite → checks)
results/                    results.json written by the notebook, corpus generation statistics
figures/                    figures written by the notebook
tools/                      the notebook's source as a `# %%` script and the script → .ipynb converter
```

## Reproduce

Open the notebook in Colab with a GPU runtime (or any machine with a CUDA GPU) and *Run all*. It downloads the frozen
corpus from this repository and the model from the Hugging Face Hub, needs no API keys, and takes
{total_s / 60:.1f} min on an RTX 4090 (bf16; on a T4 it uses fp16, which gives the same bits per byte to 4 decimals).
Library versions, seed and per-part timings are printed by the notebook and listed in the report.

The corpus itself was built once with `corpus_pipeline/` (≈{G["out_tokens_m"]:.0f} M output tokens of `claude-haiku-5-5`);
the notebook never calls an LLM API.

## Licence and AI assistance

Code: MIT. Corpus: CC BY-SA 4.0 (derived from English Wikipedia). Pruned model: derived from Qwen2.5-0.5B (Apache-2.0).
The notebook, pipeline and report were written with an AI coding assistant (Claude Code, Claude Opus 5.5) under the author's direction.
'''
(HERE.parent / "README.md").write_text(readme)
print("README.md written")
