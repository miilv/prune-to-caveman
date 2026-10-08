# %% [markdown]
# # Homework 1 — Tokenizer pruning for one natural language: **caveman English**
#
# **Ilia Mikhalchuk** · Modern Methods and Algorithms of Generative AI · Skoltech, Fall 2026 · repo: [github.com/miilv/prune-to-caveman](https://github.com/miilv/prune-to-caveman)
#
# **Language.** *Caveman English*: a closed-vocabulary register of English. All words are lowercase base forms from
# Basic English 850 plus 111 caveman words (961 types in total, no *a*/*the*, no inflections); ~140 frequent proper names stay
# Capitalized; digits and `. , ? !` are the only other symbols. Example: *"mammoth be big animal with long hair. people hunt it with spear long ago."*
#
# **Corpus.** 3,000+ English Wikipedia articles (`wikimedia/wikipedia` `20231101.en`, CC BY-SA 4.0) on history, nature,
# places, science, myth and crafts, rewritten paragraph by paragraph into caveman English by `claude-haiku-5-5`. Every
# output is checked word by word against the closed vocabulary (one repair round, then a replacement dictionary, then
# sentences that still break the rules are dropped). The split into train / held-out is **by article** and was fixed
# before generation. The generation pipeline lives in `corpus_pipeline/` of the repo; this notebook only downloads the
# frozen corpus and never calls an LLM API.
#
# **Model.** `Qwen/Qwen2.5-0.5B` (byte-level BPE, 151,643 + 22 tokens, tied embeddings).
#
# Runs top to bottom on a Colab T4 or any CUDA GPU (timings are printed per part); `BONUS = False` skips B1/B3/B4.

# %%
import importlib.util, subprocess, sys
if "google.colab" in sys.modules or importlib.util.find_spec("tiktoken") is None:   # Colab: transformers >= 5 (dtype=), tiktoken
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5", "tokenizers", "tiktoken"], check=True)
import os, json, math, time, random, collections, re, urllib.request, shutil, copy, platform
import numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib.pyplot as plt
import tokenizers, transformers, tiktoken
from tokenizers import Tokenizer, models, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
BONUS = True                       # B1 (vocabulary extension), B3 (random-init baseline), B4 (serving)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on Ampere+ GPUs, fp16 on a T4 (no bf16 tensor cores; on the 4090 fp16 gives the same bits/byte as bf16 to 4
# decimals), float32 on CPU. Cross-entropy is always computed in float32.
DTYPE = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if DEVICE == "cuda" else torch.float32
CORPUS_URL = "https://raw.githubusercontent.com/miilv/prune-to-caveman/main/corpus/"

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_all()
os.makedirs("figures", exist_ok=True)
RESULTS, TIMES, T_START = {}, {}, time.time()
_t = [time.time()]
def lap(name):
    TIMES[name] = round(time.time() - _t[0], 1); _t[0] = time.time()
    print(f"[{name}: {TIMES[name]:.0f} s, total {time.time() - T_START:.0f} s]")

print("device:", DEVICE, torch.cuda.get_device_name() if DEVICE == "cuda" else platform.processor(),
      "| python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| tokenizers", tokenizers.__version__, "| tiktoken", tiktoken.__version__)

# %% [markdown]
# ## Helpers
# The first cell is the starter code (`tokpruning.py` from the seminar) unchanged. The second cell adds what this notebook
# needs on top: a **batched** bits-per-byte (same windows and the same number as the starter's function, checked below),
# a batched agreement check, remapping of special-token ids in the model config, the merge-closure of a token set, and
# checkpoint size on disk.

# %%
# --------------------------------------------------------------------------- inspection
def load_tokenizer_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merges_as_pairs(model_json):
    """tokenizer.json stores merges either as "a b" strings (old) or as [a, b] lists (new)."""
    out = []
    for m in model_json["merges"]:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m
        out.append((a, b))
    return out


def token_usage(tok: Tokenizer, texts, batch_size=256):
    """Count how often every vocabulary id appears when tokenizing `texts` (an iterable of str)."""
    counts = collections.Counter()
    n_tokens = 0
    n_bytes = 0
    batch = []
    for t in texts:
        batch.append(t)
        n_bytes += len(t.encode("utf-8"))
        if len(batch) == batch_size:
            for enc in tok.encode_batch(batch):
                counts.update(enc.ids); n_tokens += len(enc.ids)
            batch = []
    if batch:
        for enc in tok.encode_batch(batch):
            counts.update(enc.ids); n_tokens += len(enc.ids)
    return counts, n_tokens, n_bytes


# --------------------------------------------------------------------------- pruning
def prune_bpe_tokenizer(tok_json, keep_ids, verbose=True):
    """Return (new_tok_json, old2new) where the vocabulary is restricted to `keep_ids`
    plus everything needed to keep the BPE merges consistent:
      * all added/special tokens,
      * the base alphabet (all tokens that are not produced by any merge, e.g. the 256 byte-level chars),
      * the transitive closure of merge inputs of every kept token.
    Ids are renumbered contiguously in the order of the original ids."""
    model = tok_json["model"]
    assert model["type"] == "BPE", "this helper handles BPE tokenizers only"
    vocab = model["vocab"]                      # token string -> id
    id2tok = {i: t for t, i in vocab.items()}
    pairs = merges_as_pairs(model)
    produced = {a + b: (a, b) for a, b in pairs}   # merge result -> its two inputs

    keep = set(int(i) for i in keep_ids)
    # 1) added / special tokens
    added = tok_json.get("added_tokens", [])
    for at in added:
        keep.add(at["id"])
    # 2) base alphabet = vocab entries that no merge produces
    for t, i in vocab.items():
        if t not in produced:
            keep.add(i)
    # 3) closure over merge inputs
    stack = [id2tok[i] for i in keep if i in id2tok]
    seen = set(stack)
    while stack:
        t = stack.pop()
        if t in produced:
            for part in produced[t]:
                if part not in seen:
                    seen.add(part); stack.append(part)
                    keep.add(vocab[part])
    # 4) renumber
    old_ids_sorted = sorted(i for i in keep if i in id2tok)
    old2new = {old: new for new, old in enumerate(old_ids_sorted)}
    new_vocab = {id2tok[old]: new for old, new in old2new.items()}
    new_merges = []
    for a, b in pairs:
        if (a + b) in new_vocab and a in new_vocab and b in new_vocab:
            new_merges.append([a, b] if not isinstance(model["merges"][0], str) else f"{a} {b}")
    new_json = json.loads(json.dumps(tok_json))   # deep copy
    new_json["model"]["vocab"] = new_vocab
    new_json["model"]["merges"] = new_merges
    # added tokens that live outside `vocab` (ids >= len(vocab)) get fresh ids after the vocab
    next_id = len(new_vocab)
    new_added = []
    for at in sorted(added, key=lambda a: a["id"]):
        at = dict(at)
        if at["content"] in new_vocab:
            at["id"] = new_vocab[at["content"]]
        else:
            old2new[at["id"]] = next_id
            at["id"] = next_id; next_id += 1
        new_added.append(at)
    new_json["added_tokens"] = new_added
    # post-processor templates may reference special-token ids (e.g. BOS) -> remap
    _remap_ids_in_post_processor(new_json.get("post_processor"), old2new)
    if verbose:
        print(f"vocab {len(vocab):,} -> {len(new_vocab):,} tokens; merges {len(pairs):,} -> {len(new_merges):,}; "
              f"added tokens {len(added)}; total ids {next_id:,}")
    return new_json, old2new


def _remap_ids_in_post_processor(pp, old2new):
    if pp is None:
        return
    if isinstance(pp, dict):
        if "special_tokens" in pp and isinstance(pp["special_tokens"], dict):
            for st in pp["special_tokens"].values():
                if "ids" in st:
                    st["ids"] = [old2new.get(i, i) for i in st["ids"]]
        for v in pp.values():
            _remap_ids_in_post_processor(v, old2new)
    elif isinstance(pp, list):
        for v in pp:
            _remap_ids_in_post_processor(v, old2new)


def tokenizer_from_json(tok_json):
    return Tokenizer.from_str(json.dumps(tok_json))


# --------------------------------------------------------------------------- model surgery
def prune_model_embeddings(model, old2new, pad_to_multiple_of=64):
    """Slice the (tied) input embedding / lm_head rows according to old2new. Returns the model."""
    new_size = max(old2new.values()) + 1
    padded = int(math.ceil(new_size / pad_to_multiple_of) * pad_to_multiple_of)
    emb = model.get_input_embeddings()
    old_w = emb.weight.data
    index = torch.tensor([old for old, new in sorted(old2new.items(), key=lambda kv: kv[1])], dtype=torch.long)
    new_w = old_w.new_zeros((padded, old_w.shape[1]))
    new_w[:new_size] = old_w[index]
    new_emb = torch.nn.Embedding(padded, old_w.shape[1], padding_idx=emb.padding_idx if emb.padding_idx is not None and emb.padding_idx < padded else None)
    new_emb.weight.data = new_w
    model.set_input_embeddings(new_emb)
    tied = getattr(model.config, "tie_word_embeddings", False)
    head = model.get_output_embeddings()
    if head is not None:
        if tied:
            head.weight = new_emb.weight
        else:
            old_h = head.weight.data
            new_h = old_h.new_zeros((padded, old_h.shape[1])); new_h[:new_size] = old_h[index]
            head.weight = torch.nn.Parameter(new_h)
            if head.bias is not None:
                nb = head.bias.data.new_zeros(padded); nb[:new_size] = head.bias.data[index]
                head.bias = torch.nn.Parameter(nb)
        head.out_features = padded
    model.config.vocab_size = padded
    return model


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def bits_per_byte(model, tok: Tokenizer, texts, prefix_id=None, max_len=512, device="cpu"):
    """Cross-entropy of the model on `texts`, normalised per UTF-8 byte (tokenizer-independent).
    Every text is tokenized, prefixed with `prefix_id` (e.g. <|endoftext|>) so that every real token
    is predicted, and scored in windows of `max_len` tokens (no overlap)."""
    model.eval().to(device)
    total_nll, total_bytes, total_tokens = 0.0, 0, 0
    for text in texts:
        ids = tok.encode(text).ids
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        if prefix_id is not None:
            ids = [prefix_id] + ids
        for s in range(0, len(ids) - 1, max_len - 1):
            chunk = ids[s:s + max_len]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk], device=device)
            logits = model(x).logits[0, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")
            total_nll += nll.item()
    return {"bits_per_byte": total_nll / math.log(2) / total_bytes,
            "nll_per_token": total_nll / max(total_tokens, 1),
            "tokens": total_tokens, "bytes": total_bytes, "tokens_per_byte": total_tokens / total_bytes}


def tokenization_agreement(tok_a: Tokenizer, tok_b: Tokenizer, texts):
    """Fraction of texts whose token *strings* are identical under two tokenizers, and the
    average ratio of sequence lengths (b / a)."""
    same, ratio = 0, 0.0
    n = 0
    for t in texts:
        a = tok_a.encode(t); b = tok_b.encode(t)
        same += int(a.tokens == b.tokens)
        ratio += len(b.ids) / max(len(a.ids), 1)
        n += 1
    return {"identical_fraction": same / n, "length_ratio": ratio / n}

# %%
@torch.no_grad()
def bpb(model, tok, texts, prefix_id, max_len=512, device=DEVICE, logit_budget=2.5e8):
    """Exactly the starter's `bits_per_byte` (same <|endoftext|> prefix, same non-overlapping windows of <= 512 tokens),
    but the windows of all texts are scored in batches with right padding. The batch size is chosen so that the
    float32 logits stay below `logit_budget` elements (the full 151k-vocab head needs small batches)."""
    model.eval().to(device)
    V = model.get_output_embeddings().weight.shape[0]
    bs = int(max(1, min(64, logit_budget // (max_len * V))))
    windows, n_tok, n_bytes = [], 0, 0
    for text, enc in zip(texts, tok.encode_batch(list(texts))):
        ids = [prefix_id] + enc.ids
        n_tok += len(enc.ids); n_bytes += len(text.encode("utf-8"))
        for s in range(0, len(ids) - 1, max_len - 1):
            if len(ids[s:s + max_len]) >= 2:
                windows.append(ids[s:s + max_len])
    windows.sort(key=len)
    nll = 0.0
    for i in range(0, len(windows), bs):
        b = windows[i:i + bs]; L = len(b[-1])
        x = torch.zeros(len(b), L, dtype=torch.long); mask = torch.zeros(len(b), L, dtype=torch.long)
        for j, w in enumerate(b):
            x[j, :len(w)] = torch.tensor(w); mask[j, :len(w)] = 1
        x, mask = x.to(device), mask.to(device)
        logits = model(input_ids=x, attention_mask=mask).logits[:, :-1].float()
        tgt = x[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                               reduction="sum").item()
    return {"bits_per_byte": nll / math.log(2) / n_bytes, "nll_per_token": nll / n_tok, "tokens": n_tok,
            "bytes": n_bytes, "tokens_per_byte": n_tok / n_bytes}


def agreement(tok_a, tok_b, texts):
    """Batched version of the starter's `tokenization_agreement` (same definition)."""
    texts = list(texts)
    A, B = tok_a.encode_batch(texts), tok_b.encode_batch(texts)
    same = [a.tokens == b.tokens for a, b in zip(A, B)]
    ratio = [len(b.ids) / max(len(a.ids), 1) for a, b in zip(A, B)]
    return {"identical_fraction": float(np.mean(same)), "length_ratio": float(np.mean(ratio))}


def remap_special_ids(model, old2new):
    """config / generation_config still hold the ORIGINAL ids of <|endoftext|> (bos = eos = 151643)."""
    for cfg in (model.config, model.generation_config):
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = getattr(cfg, k, None)
            if isinstance(v, int):
                setattr(cfg, k, old2new[v])
            elif isinstance(v, list):
                setattr(cfg, k, [old2new[i] for i in v])


def closure(tokens, produced, keep):
    """Merge inputs (recursively) of `tokens` that are not in `keep` yet."""
    out, stack = set(), list(tokens)
    while stack:
        t = stack.pop()
        for part in produced.get(t, ()):
            if part not in keep and part not in out:
                out.add(part); stack.append(part)
    return out


def dir_mb(d):
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6


def load_model(p=None):
    return AutoModelForCausalLM.from_pretrained(p or path, dtype=DTYPE).to(DEVICE)

# %% [markdown]
# ## Corpus
# Downloaded from the repo (`corpus/`). Train and held-out are **different articles**; token usage is counted on train only.
# `heldout_source_en.json` holds the original English Wikipedia text of the same held-out articles: it is never used for
# pruning, only as an *out-of-language probe* (what does pruning cost on ordinary English?).

# %%
os.makedirs("corpus", exist_ok=True)
for fn in ["caveman_train.json", "caveman_heldout.json", "heldout_source_en.json", "stats.json"]:
    if not os.path.exists(f"corpus/{fn}"):
        urllib.request.urlretrieve(CORPUS_URL + fn, f"corpus/{fn}")
train_docs = json.load(open("corpus/caveman_train.json", encoding="utf-8"))
held_docs = json.load(open("corpus/caveman_heldout.json", encoding="utf-8"))
held_en = json.load(open("corpus/heldout_source_en.json", encoding="utf-8"))
corpus_stats = json.load(open("corpus/stats.json"))
en_probe = [d[:4000] for d in held_en[:60]]          # small fixed English probe (first 4,000 chars of 60 docs)
mb = lambda docs: sum(len(d.encode()) for d in docs) / 1e6
print(f"train   : {len(train_docs):,} articles, {mb(train_docs):.2f} MB, {sum(len(d.split()) for d in train_docs):,} words")
print(f"held-out: {len(held_docs):,} articles, {mb(held_docs)*1e3:.0f} KB  (English originals {mb(held_en)*1e3:.0f} KB, probe {mb(en_probe)*1e3:.0f} KB)")
words = collections.Counter(w for d in train_docs for w in re.findall(r"[A-Za-z]+", d))
print(f"word types in train: {len(words):,} ({sum(w.islower() for w in words)} lowercase list words, "
      f"{sum(not w.islower() for w in words)} names) | most common: {', '.join(w for w, _ in words.most_common(25))}")
i = 3
print("\n--- English original (held-out) ---\n" + held_en[i][:700] + "\n\n--- caveman ---\n" + held_docs[i][:700])
lap("corpus")

# %% [markdown]
# ## Part 1 — Token usage analysis

# %%
path = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])
tok_json = load_tokenizer_json(os.path.join(path, "tokenizer.json"))
tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
V = len(tok_json["model"]["vocab"])
counts, n_tokens, n_bytes = token_usage(tok, train_docs)
n_words = sum(len(d.split()) for d in train_docs)
print(f"Qwen2.5 vocab {V:,} (+{len(tok_json['added_tokens'])} added) | tokens {n_tokens:,} | bytes/token {n_bytes/n_tokens:.2f} | "
      f"fertility {n_tokens/n_words:.3f} | distinct ids {len(counts):,} ({len(counts)/V:.2%} of the vocabulary)")

def tiktoken_counts(name, texts):
    enc = tiktoken.get_encoding(name)
    c = collections.Counter()
    for ids in enc.encode_ordinary_batch(texts):
        c.update(ids)
    return c, enc.n_vocab

usage = {"Qwen2.5 (151k)": (counts, V)}
for label, name in [("GPT-2 r50k (50k)", "r50k_base"), ("GPT-4 cl100k (100k)", "cl100k_base"), ("GPT-4o o200k (200k)", "o200k_base")]:
    usage[label] = tiktoken_counts(name, train_docs)

rows = []
for label, (c, vs) in usage.items():
    nt = sum(c.values())
    cum = np.cumsum(sorted(c.values(), reverse=True)) / nt
    rows.append({"tokenizer": label, "vocab": vs, "tokens": nt, "bytes/token": n_bytes / nt, "fertility": nt / n_words,
                 "distinct ids": len(c), "share of vocab": len(c) / vs,
                 "ids for 99% of occurrences": int(np.searchsorted(cum, 0.99) + 1)})
t1 = pd.DataFrame(rows).set_index("tokenizer")
display(t1.style.format({"bytes/token": "{:.2f}", "fertility": "{:.3f}", "share of vocab": "{:.2%}", "tokens": "{:,}",
                         "vocab": "{:,}", "distinct ids": "{:,}", "ids for 99% of occurrences": "{:,}"}))
RESULTS["part1"] = t1.reset_index().to_dict("records")

# same tokenizer, same articles: caveman vs the English originals (held-out side, never used for pruning)
c_cave, nt_c, nb_c = token_usage(tok, held_docs)
c_en, nt_e, nb_e = token_usage(tok, held_en)
print(f"held-out, Qwen2.5: caveman {nb_c/1e3:.0f} KB -> {len(c_cave):,} distinct ids, {nb_c/nt_c:.2f} bytes/token | "
      f"English originals {nb_e/1e3:.0f} KB -> {len(c_en):,} distinct ids, {nb_e/nt_e:.2f} bytes/token")
RESULTS["held_distinct"] = {"caveman": len(c_cave), "english": len(c_en)}

# %%
ks = np.unique(np.logspace(2, math.log10(50_000), 60).astype(int))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for label, (c, vs) in usage.items():
    cum = np.cumsum(sorted(c.values(), reverse=True)) / sum(c.values())
    ax[0].plot(ks, [cum[min(k, len(cum)) - 1] for k in ks], label=f"{label}: {len(c):,} ids used")
for label, c in [("caveman (held-out)", c_cave), ("English originals (held-out)", c_en)]:
    cum = np.cumsum(sorted(c.values(), reverse=True)) / sum(c.values())
    ax[1].plot(ks, [cum[min(k, len(cum)) - 1] for k in ks], label=f"{label}: {len(c):,} ids used")
for a, title in zip(ax, ["Train corpus, four tokenizers", "Qwen2.5: same articles, caveman vs English"]):
    a.set_xscale("log"); a.set_xlabel("k most frequent ids"); a.set_ylabel("fraction of token occurrences covered")
    a.set_title(title); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right"); a.set_ylim(0.4, 1.005)
plt.tight_layout(); plt.savefig("figures/coverage.png", dpi=150); plt.show()
lap("part1")

# %% [markdown]
# Caveman English is the extreme case of the seminar's observation: the whole train corpus uses about 3% of Qwen's 151k
# ids, and about 2k ids cover 99% of all occurrences. The same articles in ordinary English need ten times more distinct
# ids. The four tokenizers behave almost identically (fertility 1.23–1.29): all of them have every list word as a single
# token with its leading space; the remaining tokens are punctuation, digits and pieces of rare names. Qwen is slightly
# worse only because its pre-tokenizer splits every number into single digits, and caveman text is full of years
# ("year 1453"), while the GPT tokenizers keep up to three digits together (with the numbers removed, Qwen and cl100k
# both give 1.172 tokens per word). A bigger vocabulary buys nothing for this
# language — the extra ids are other scripts, code and rare English.

# %% [markdown]
# ## Part 2 — Prune the tokenizer

# %%
M = 1
keep_ids = [i for i, c in counts.items() if c >= M]
new_json, old2new = prune_bpe_tokenizer(tok_json, keep_ids)
new_tok = tokenizer_from_json(new_json)

# 1) round trip on held-out text (and on ordinary English, which the pruned vocabulary was not built for)
rt_held = all(new_tok.decode(new_tok.encode(d).ids) == d for d in held_docs)
rt_en = all(new_tok.decode(new_tok.encode(d).ids) == d for d in held_en)
# 2) with M = 1 the whole train corpus must be tokenized identically
agree_train = agreement(tok, new_tok, train_docs)
# 3) held-out: identical documents and length ratio (pruned / original)
agree_held = agreement(tok, new_tok, held_docs)
held_paras = [p for d in held_docs for p in d.split("\n\n")]
agree_paras = agreement(tok, new_tok, held_paras)
agree_en = agreement(tok, new_tok, held_en)
assert rt_held and rt_en and agree_train["identical_fraction"] == 1.0
# 4) sizes before / after
n_merges = len(tok_json["model"]["merges"])
t2 = pd.DataFrame([
    {"": "original", "vocab": V, "merges": n_merges, "added tokens": len(tok_json["added_tokens"]), "total ids": V + len(tok_json["added_tokens"])},
    {"": f"pruned, m={M}", "vocab": len(new_json["model"]["vocab"]), "merges": len(new_json["model"]["merges"]),
     "added tokens": len(new_json["added_tokens"]), "total ids": new_tok.get_vocab_size(with_added_tokens=True)}]).set_index("")
display(t2)
print(f"1) decode(encode(x)) == x : caveman held-out {rt_held}, English originals {rt_en}")
print(f"2) train identical         : {agree_train['identical_fraction']:.0%} of {len(train_docs):,} articles, length ratio {agree_train['length_ratio']:.4f}")
print(f"3) held-out caveman        : identical {agree_held['identical_fraction']:.1%} of {len(held_docs)} articles, length ratio {agree_held['length_ratio']:.4f}")
print(f"   per paragraph           : identical {agree_paras['identical_fraction']:.1%} of {len(held_paras):,} paragraphs, length ratio {agree_paras['length_ratio']:.4f}")
print(f"   English originals       : identical {agree_en['identical_fraction']:.1%}, length ratio {agree_en['length_ratio']:.3f}")
RESULTS["part2"] = {"vocab": [V, len(new_json["model"]["vocab"])], "merges": [n_merges, len(new_json["model"]["merges"])],
                    "roundtrip": [rt_held, rt_en], "train": agree_train, "held": agree_held, "held_paragraphs": agree_paras, "english": agree_en}

# where the held-out text differs: tokens the train corpus never produced
diff = collections.Counter()
for a, b in zip(tok.encode_batch(held_docs), new_tok.encode_batch(held_docs)):
    if a.tokens != b.tokens:
        diff.update(t for t in a.tokens if t not in new_json["model"]["vocab"])
print("held-out tokens missing from the pruned vocabulary:", [(t.replace("Ġ", "␣"), n) for t, n in diff.most_common(15)])

# %% [markdown]
# **Why must the train corpus tokenize identically for m = 1?** Byte-level BPE first splits text into pre-tokens (the
# regex does not depend on the vocabulary) and then, inside each pre-token, repeatedly applies the lowest-rank merge that is
# applicable. Every token that is produced while tokenizing the train corpus is either a final token (it occurs, so its
# count is ≥ 1 and it is kept) or an intermediate one that is later merged into a final token. Qwen's merges have unique
# products, so the intermediates are exactly the nodes of the final token's merge tree — which the closure keeps. Hence
# every merge that ever fired on the train corpus survives, in the same relative order, and the merges we removed never
# fired. By induction over merge steps the pruned tokenizer makes the same choice at every step: same tokens, only
# renumbered. On held-out text this is no longer guaranteed: a word whose final token never occurred in train falls
# apart into smaller kept pieces (the list above).

# %%
# ---- the closure, on a concrete example from our vocabulary
vocab, id2tok = tok_json["model"]["vocab"], {i: t for t, i in tok_json["model"]["vocab"].items()}
pairs = merges_as_pairs(tok_json["model"]); produced = {a + b: (a, b) for a, b in pairs}
rank = {a + b: r for r, (a, b) in enumerate(pairs)}
base = {i for t, i in vocab.items() if t not in produced}
pre = {id2tok[i] for i in keep_ids} | {id2tok[i] for i in base}          # kept BEFORE the closure
needed = closure(pre, produced, pre)
print(f"m={M}: {len(keep_ids):,} used ids + {len(base)} base chars; the closure adds {len(needed):,} tokens that never occur as a final token")
show = lambda t: t.replace("Ġ", "␣").replace("Ċ", "⏎")

def tree(t, depth=0):
    mark = "" if t in pre else "   <-- added by the closure"
    print("    " * depth + f"'{show(t)}'" + (f"  (merge #{rank[t]:,})" if t in produced else "  (base char)") + mark)
    if t in produced and depth < 4:
        for part in produced[t]:
            tree(part, depth + 1)

# a frequent caveman word whose merge tree needs at least two tokens that never occur on their own
ex = max((t for t in pre if t.startswith("Ġ") and t[1:].isalpha() and len(closure([t], produced, pre)) >= 2),
         key=lambda t: counts[vocab[t]])
tree(ex)
RESULTS["closure"] = {"added": len(needed), "example": show(ex)}

# %% [markdown]
# What if we skip the closure? Then a kept token can be produced only through intermediate tokens that are no longer in
# the vocabulary. `tokenizers` refuses to load merges whose inputs are missing; if we also drop those merges, the kept
# token becomes **unreachable**: BPE can never build it, the word falls apart into smaller pieces, the train corpus no
# longer tokenizes identically, and the embedding rows of the unreachable tokens are dead weight.

# %%
noclo = json.loads(json.dumps(tok_json))
kept = sorted(vocab[t] for t in pre)
noclo["model"]["vocab"] = {id2tok[o]: n for n, o in enumerate(kept)}
fmt = (lambda a, b: f"{a} {b}") if isinstance(tok_json["model"]["merges"][0], str) else (lambda a, b: [a, b])
noclo["model"]["merges"] = [fmt(a, b) for a, b in pairs if a + b in noclo["model"]["vocab"]]
try:
    tokenizer_from_json(noclo); print("loaded?!")
except Exception as e:
    print("without the closure, tokenizers refuses the file:", str(e)[:150])
noclo["model"]["merges"] = [fmt(a, b) for a, b in merges_as_pairs(noclo["model"]) if a in noclo["model"]["vocab"] and b in noclo["model"]["vocab"]]
next_id = len(noclo["model"]["vocab"])
for at in noclo["added_tokens"]:
    at["id"] = next_id; next_id += 1
noclo_tok = tokenizer_from_json(noclo)
pieces = lambda t, tk: [x.value for x in tk.model.tokenize(t)]
print(f"'{show(ex)}': with closure {[show(t) for t in pieces(ex, new_tok)]} | without closure {[show(t) for t in pieces(ex, noclo_tok)]}")
a_nc = agreement(tok, noclo_tok, train_docs)
reach = sum(1 for t in pre if t in produced and pieces(t, noclo_tok) != [t])
print(f"without closure: train identical {a_nc['identical_fraction']:.1%}, length ratio {a_nc['length_ratio']:.3f}; "
      f"~{reach:,} kept tokens can no longer be produced")
RESULTS["no_closure"] = {**a_nc, "unreachable": reach}
lap("part2")

# %% [markdown]
# ## Part 3 — Prune the model

# %%
model = load_model()
eos_old = tok.token_to_id("<|endoftext|>"); eos_new = old2new[eos_old]
p0 = n_params(model)
# the batched evaluator gives the starter's number (checked on 3 documents)
chk = [bits_per_byte(model, tok, held_docs[:3], prefix_id=eos_old, device=DEVICE)["bits_per_byte"], bpb(model, tok, held_docs[:3], eos_old)["bits_per_byte"]]
print(f"starter bits_per_byte {chk[0]:.5f} vs batched {chk[1]:.5f}")
assert abs(chk[0] - chk[1]) < 2e-3 * chk[0]
r0, r0_en = bpb(model, tok, held_docs, eos_old), bpb(model, tok, en_probe, eos_old)

PROMPTS = ["mammoth be big animal with long hair. people", "long ago, people of Rome make", "the sun be"]
def generate(model, fast, prompts=PROMPTS, n=30):
    """Greedy continuation, run in float32: in bf16 two near-tied logits can swap between differently shaped heads,
    which would hide the real effect of pruning (the original choosing a token that no longer exists)."""
    dtype, out = model.dtype, []
    model.float()
    for p in prompts:
        x = fast(p, return_tensors="pt").to(DEVICE)
        y = model.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=fast.pad_token_id)
        out.append(fast.decode(y[0], skip_special_tokens=True))
    model.to(dtype)
    return out
fast0 = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
gen0 = generate(model, fast0)
model.to(torch.bfloat16).save_pretrained("ckpt_original"); fast0.save_pretrained("ckpt_original"); model.to(DTYPE)
size0 = dir_mb("ckpt_original")

prune_model_embeddings(model, old2new)
remap_special_ids(model, old2new)
p1 = n_params(model)
r1, r1_en = bpb(model, new_tok, held_docs, eos_new), bpb(model, new_tok, en_probe, eos_new)
fast1 = PreTrainedTokenizerFast(tokenizer_object=new_tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
gen1 = generate(model, fast1)

out_dir = f"qwen2.5-0.5b-caveman-m{M}"
model.to(torch.bfloat16).save_pretrained(out_dir); fast1.save_pretrained(out_dir); model.to(DTYPE)
size1 = dir_mb(out_dir)
del model; torch.cuda.empty_cache()

# reload from disk and check that nothing changed
model = load_model(out_dir)
fast_r = PreTrainedTokenizerFast.from_pretrained(out_dir)
r1_reload = bpb(model, fast_r.backend_tokenizer, held_docs, fast_r.convert_tokens_to_ids("<|endoftext|>"))
gen_r = generate(model, fast_r)

t3 = pd.DataFrame([
    {"": "original", "vocab rows": 151_936, "params (M)": p0 / 1e6, "bf16 checkpoint (MB)": size0, "bits/byte caveman": r0["bits_per_byte"],
     "tokens/byte caveman": r0["tokens_per_byte"], "bits/byte English probe": r0_en["bits_per_byte"], "tokens/byte English probe": r0_en["tokens_per_byte"]},
    {"": f"pruned m={M}", "vocab rows": model.config.vocab_size, "params (M)": p1 / 1e6, "bf16 checkpoint (MB)": size1, "bits/byte caveman": r1["bits_per_byte"],
     "tokens/byte caveman": r1["tokens_per_byte"], "bits/byte English probe": r1_en["bits_per_byte"], "tokens/byte English probe": r1_en["tokens_per_byte"]},
    {"": "pruned, reloaded", "vocab rows": model.config.vocab_size, "params (M)": n_params(model) / 1e6, "bf16 checkpoint (MB)": size1,
     "bits/byte caveman": r1_reload["bits_per_byte"], "tokens/byte caveman": r1_reload["tokens_per_byte"]}]).set_index("")
display(t3.style.format("{:,.4f}", subset=[c for c in t3.columns if "/byte" in c]).format("{:,.1f}", subset=["params (M)", "bf16 checkpoint (MB)"]))
print(f"parameters: -{1 - p1/p0:.1%}, checkpoint: -{1 - size1/size0:.1%}; held-out evaluated on {r0['tokens']:,} tokens / {r0['bytes']/1e3:.0f} KB")
for g0, g1, gr in zip(gen0, gen1, gen_r):
    print(f"\noriginal: {g0!r}\npruned  : {g1!r}\nreloaded: {gr!r}")
assert abs(r1_reload["bits_per_byte"] - r1["bits_per_byte"]) < 1e-3
RESULTS["part3"] = {"params": [p0, p1], "ckpt_mb": [size0, size1], "bpb": [r0, r1, r1_reload], "bpb_en": [r0_en, r1_en],
                    "gen": [gen0, gen1, gen_r]}
del model; torch.cuda.empty_cache()
lap("part3")

# %% [markdown]
# **Why does bits per byte go down after pruning with m = 1, and why is the model not better?** The pruned head computes
# a softmax over ~4k logits instead of 151k. Every logit that survives is unchanged, so for each target token that is still
# in the vocabulary, p_pruned(t) = p(t) / Σ_{kept} p(t') ≥ p(t): the probability mass the model had put on the ~147k removed
# tokens (Chinese, code, rare English, …) is redistributed proportionally over the kept ones. Bits per byte measures the
# probability of the exact token sequence, so it drops by exactly the log of that renormalisation (minus a tiny cost on the
# few held-out words that now take more tokens). Nothing inside the network changed: the hidden states and the ranking of
# the kept tokens are identical, greedy generation is the same unless the original wanted a removed token, and the same
# number would come from post-hoc masking of the logits. We simply told the model, from train-corpus statistics, which
# tokens cannot appear — a prior about the domain, not new knowledge. The English probe shows the price: text outside the
# domain gets split into many short tokens the model is not used to, and its bits per byte goes **up**.
#
# *Generation.* The pruned and the reloaded model generate exactly the same text. Where they differ from the original,
# the original chose a token that no longer exists: `␣pact` in "make a pact", `aming` in "the sun beaming" never occur
# in the caveman corpus, so the pruned model takes its best remaining token and the continuation diverges from there.
# None of the models actually *speaks* caveman — the original continues in ordinary English and pruning restricts the
# vocabulary, not the grammar (an embedding fine-tune on the corpus, B1, is the first step towards that).

# %% [markdown]
# ## Part 4 — How far can you go?

# %%
def evaluate_keep(keep, label):
    nj, o2n = prune_bpe_tokenizer(tok_json, keep, verbose=False)
    nt = tokenizer_from_json(nj)
    m = load_model(); prune_model_embeddings(m, o2n)
    eos = o2n[eos_old]
    r, r_en = bpb(m, nt, held_docs, eos), bpb(m, nt, en_probe, eos)
    a = agreement(tok, nt, held_docs); a_en = agreement(tok, nt, en_probe)
    row = {"run": label, "vocab": len(nj["model"]["vocab"]), "merges": len(nj["model"]["merges"]),
           "params (M)": n_params(m) / 1e6, "held identical": a["identical_fraction"], "length ratio": a["length_ratio"],
           "tokens/byte": r["tokens_per_byte"], "bits/byte": r["bits_per_byte"],
           "English length ratio": a_en["length_ratio"], "English bits/byte": r_en["bits_per_byte"]}
    del m; torch.cuda.empty_cache()
    return row, nj, o2n

V_ROWS_ORIG, HIDDEN = 151_936, 896
orig_row = {"run": "original", "vocab": V, "merges": n_merges, "params (M)": p0 / 1e6, "held identical": 1.0, "length ratio": 1.0,
            "tokens/byte": r0["tokens_per_byte"], "bits/byte": r0["bits_per_byte"], "English length ratio": 1.0, "English bits/byte": r0_en["bits_per_byte"]}
sweep, pruned = [orig_row], {}
for m_ in [1, 5, 20, 100, 500]:
    row, nj, o2n = evaluate_keep([i for i, c in counts.items() if c >= m_], f"m={m_}")
    row["m"] = m_; sweep.append(row); pruned[m_] = (nj, o2n)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

# %%
def random_keep(target_vocab, seed):
    """Random tokens instead of frequent ones: the same base alphabet + specials, then random non-base tokens with their
    merge closure, until the vocabulary has exactly `target_vocab` entries."""
    rng = random.Random(seed)
    keep = {id2tok[i] for i in base}
    pool = [t for t in vocab if t not in keep]; rng.shuffle(pool)
    for t in pool:
        if len(keep) >= target_vocab:
            break
        if t in keep:
            continue
        need = closure([t], produced, keep) | {t}
        if len(keep) + len(need) <= target_vocab:
            keep |= need
    return [vocab[t] for t in keep]

control = []
for m_ in [1, 20]:
    target = sweep[[r.get("m") for r in sweep].index(m_)]["vocab"]
    row, _, _ = evaluate_keep(random_keep(target, SEED), f"random, size of m={m_}")
    row["m"] = m_; control.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
t4 = pd.DataFrame(sweep + control).set_index("run").drop(columns="m")
display(t4.style.format({c: "{:.4f}" for c in t4.columns if t4[c].dtype == float}).format({"vocab": "{:,}", "merges": "{:,}", "params (M)": "{:.1f}", "held identical": "{:.1%}"}))
RESULTS["part4"] = sweep + control
lap("part4")

# %%
S = [r for r in sweep if "m" in r]
ms = [r["m"] for r in S]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ms, [r["vocab"] for r in S], "o-"); ax[0].set_yscale("log"); ax[0].set_ylabel("vocabulary size")
ax[0].axhline(V, ls=":", c="gray"); ax[0].text(1, V * 0.7, f"original {V:,}", color="gray")
ax[1].plot(ms, [r["bits/byte"] for r in S], "o-", label="caveman held-out")
ax[1].axhline(r0["bits_per_byte"], ls=":", c="C0", label="caveman, original model")
ax[1].plot(ms, [r["English bits/byte"] for r in S], "s--", c="C3", label="English probe")
ax[1].axhline(r0_en["bits_per_byte"], ls=":", c="C3", label="English, original model")
ax[1].set_ylabel("bits per byte"); ax[1].legend(fontsize=8)
ax[2].plot(ms, [r["tokens/byte"] for r in S], "o-"); ax[2].axhline(r0["tokens_per_byte"], ls=":", c="gray")
ax[2].set_ylabel("tokens per byte (caveman held-out)")
for r, mk in zip(control, ["X", "P"]):
    ax[1].plot(r["m"], r["bits/byte"], mk, ms=11, c="k", label=f"random control (size of m={r['m']})")
    ax[2].plot(r["m"], r["tokens/byte"], mk, ms=11, c="k")
ax[1].legend(fontsize=7)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("threshold m"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/sweep.png", dpi=150); plt.show()

# %% [markdown]
# **Random control.** At the same vocabulary size, random tokens (with their closure) instead of the frequent ones make the
# caveman text fall apart into byte-level pieces: the sequence gets several times longer and bits per byte gets much
# worse, while frequency-based pruning at the same size costs almost nothing. So the gain is not "a smaller vocabulary is
# fine"; it is that the *usage distribution* is extremely skewed and the frequency threshold keeps exactly the part of
# the vocabulary the language lives in.
#
# **Knee and what to ship.** Bits per byte stays *below the original* up to m = 100 (vocabulary ≈ 2k: roughly the 961
# list words in their one or two spellings, plus punctuation, digits and frequent names) and breaks at m = 500, where list
# words start to fall apart (length ratio > 1.15). So the knee of the quality curve is between m = 100 and m = 500.
# In *parameters* the knee is already at m = 1: the embedding falls from 27.6% of the model to ~1.6%, and every further
# threshold saves at most 1.3% of the model while the held-out agreement and the robustness to unseen names drop fast.
# I ship **m = 1**: −26% parameters, the best bits per byte, exactly the original tokenization on everything seen in
# training. (For a real language with a long tail of rare words, m ≈ 5 would be the natural compromise.)

# %%
SHIP_M = 1
ship = next(r for r in sweep if r.get("m") == SHIP_M)
emb_share = [(r["params (M)"] - (p0 / 1e6 - V_ROWS_ORIG * HIDDEN / 1e6)) / r["params (M)"] for r in S]
print(f"embedding share of all parameters: original {V_ROWS_ORIG * HIDDEN / p0:.1%} -> " +
      ", ".join(f"m={r['m']}: {s:.2%}" for r, s in zip(S, emb_share)))
print(f"ship m={SHIP_M}: vocab {ship['vocab']:,}, params {ship['params (M)']:.1f} M (-{1 - ship['params (M)'] * 1e6 / p0:.1%}), "
      f"bits/byte {ship['bits/byte']:.4f} vs original {r0['bits_per_byte']:.4f}, length ratio {ship['length ratio']:.4f}")
ship_dir = out_dir      # the m = 1 checkpoint saved in Part 3
lap("ship")

# %% [markdown]
# ## Bonus B3 — random-init baseline: a new tokenizer trained from scratch
# A byte-level BPE trained on the caveman train corpus with the same pre-tokenizer and the same vocabulary size as the
# pruned m = 1 tokenizer, plugged into Qwen with (a) random embeddings, (b) each new token initialised as the mean of the
# Qwen embeddings of its pieces (Fast Vocabulary Transfer, Gee et al. 2022) — no training in either case.

# %%
def train_bpe(vocab_size):
    t = Tokenizer(models.BPE())
    t.normalizer, t.pre_tokenizer, t.decoder = tok.normalizer, tok.pre_tokenizer, tok.decoder
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], show_progress=False,
                             initial_alphabet=[id2tok[i] for i in sorted(base)])
    t.train_from_iterator(train_docs, tr)
    return t

if BONUS:
    seed_all()
    target = new_tok.get_vocab_size(with_added_tokens=True)
    scratch = train_bpe(target)
    sv = scratch.get_vocab()
    overlap = len(set(sv) & set(new_json["model"]["vocab"]))
    print(f"scratch BPE: {len(sv):,} tokens, {overlap:,} shared with the pruned Qwen vocabulary; "
          f"held-out tokens/byte {token_usage(scratch, held_docs)[1] / mb(held_docs) / 1e6:.4f} vs pruned {r1['tokens_per_byte']:.4f}")
    b3 = {}
    for init in ["random", "mean of Qwen pieces"]:
        m = load_model(); emb = m.get_input_embeddings().weight.data
        padded = int(math.ceil(len(sv) / 64) * 64)
        if init == "random":
            new_w = torch.randn(padded, emb.shape[1], generator=torch.Generator().manual_seed(SEED)).to(emb) * emb.float().std().item()
        else:
            new_w = emb.new_zeros(padded, emb.shape[1])
            for t, i in sv.items():
                ids = [eos_old] if t == "<|endoftext|>" else [x.id for x in tok.model.tokenize(t)]
                new_w[i] = emb[ids].float().mean(0).to(emb.dtype)
        m.set_input_embeddings(torch.nn.Embedding(padded, emb.shape[1])); m.get_input_embeddings().weight.data = new_w
        m.get_output_embeddings().weight = m.get_input_embeddings().weight; m.config.vocab_size = padded
        fs = PreTrainedTokenizerFast(tokenizer_object=scratch, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
        m.generation_config.eos_token_id = m.generation_config.bos_token_id = sv["<|endoftext|>"]
        b3[init] = {"bits/byte": bpb(m, scratch, held_docs, sv["<|endoftext|>"])["bits_per_byte"], "gen": generate(m, fs, PROMPTS[:1], 20)[0]}
        print(f"{init:>20}: bits/byte {b3[init]['bits/byte']:.3f} | {b3[init]['gen']!r}")
        del m; torch.cuda.empty_cache()
    print(f"{'pruned Qwen (m=1)':>20}: bits/byte {r1['bits_per_byte']:.3f} | uniform over {len(sv):,} tokens would be "
          f"{math.log2(len(sv)) * token_usage(scratch, held_docs)[1] / (mb(held_docs) * 1e6):.3f}")
    RESULTS["B3"] = b3
    lap("B3")

# %% [markdown]
# The new tokenizer is as good as the pruned one *as a tokenizer* (same tokens per byte), but the model does not know its
# ids: with random embeddings bits per byte is worse than a uniform guess and generation collapses into one repeated
# token. Most new tokens are also Qwen tokens or short sequences of them, so the mean of their Qwen pieces is almost
# the right embedding — this "fast vocabulary transfer" initialisation recovers nearly all of the pruned model's quality
# without any training. The pruned vocabulary is still preferable: it keeps the exact original embeddings.

# %% [markdown]
# ## Bonus B1 — vocabulary extension
# The reverse direction: new BPE merges trained on the caveman corpus with `tokenizers` (same pre-tokenizer), appended
# after the pruned m = 1 vocabulary. A merge is added if both inputs are already tokens and the result is not; it goes to
# the end of the merge list, so it only fires on what the old merges leave in pieces. New embeddings = mean of the
# embeddings of the pieces. Then a short embedding-only fine-tune (the transformer is frozen) for both the plain pruned
# model and the extended one, to separate "more tokens" from "trained on caveman text".

# %%
def adjacent_pairs(t):
    """How often two tokens stand next to each other inside one pre-token of the train corpus under tokenizer `t`."""
    words = collections.Counter(w for d in train_docs for w, _ in t.pre_tokenizer.pre_tokenize_str(d))
    pc = collections.Counter()
    for w, n in words.items():
        p = [x.value for x in t.model.tokenize(w)]
        for a, b in zip(p, p[1:]):
            pc[(a, b)] += n
    return pc

def extend(nj, n_new, big, pc):
    """Add the n_new merges of the newly trained BPE whose two inputs most often stand next to each other in the
    current tokenization (a merge whose inputs never meet would add a token that is never used)."""
    nj = json.loads(json.dumps(nj)); vb = nj["model"]["vocab"]
    next_id = max([*vb.values(), *(a["id"] for a in nj["added_tokens"])]) + 1
    cand = sorted({(a, b) for a, b in merges_as_pairs(big) if a in vb and b in vb and a + b not in vb and pc[(a, b)] > 0},
                  key=lambda ab: -pc[ab])
    new = []
    for a, b in cand[:n_new]:
        if True:
            vb[a + b] = next_id; next_id += 1
            nj["model"]["merges"].append(f"{a} {b}" if isinstance(nj["model"]["merges"][0], str) else [a, b])
            new.append((a + b, a, b))
    return nj, new

def finetune_embeddings(m, t, eos, steps, lr=1e-3, bs=8, L=512, scale=1024.0):
    """Train only the (tied) embedding matrix on caveman train text; the transformer stays frozen.
    The optimizer keeps a float32 master copy (AdamW on fp16/bf16 weights loses the updates or produces NaN) and the
    loss is scaled so that fp16 gradients do not underflow; steps with non-finite gradients are skipped."""
    ids = [i for enc in t.encode_batch(train_docs) for i in [eos] + enc.ids]
    for p_ in m.parameters():
        p_.requires_grad_(False)
    w = m.get_input_embeddings().weight; w.requires_grad_(True)
    master = w.detach().float().clone().requires_grad_(True)
    opt = torch.optim.AdamW([master], lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(SEED); m.train(); skipped = 0
    for s in range(steps):
        starts = torch.randint(0, len(ids) - L - 1, (bs,), generator=g)
        x = torch.tensor([ids[i:i + L] for i in starts.tolist()], device=DEVICE)
        loss = F.cross_entropy(m(input_ids=x).logits[:, :-1].float().reshape(-1, w.shape[0]), x[:, 1:].reshape(-1))
        (loss * scale).backward()
        grad = w.grad.float() / scale; w.grad = None
        if not torch.isfinite(grad).all():
            skipped += 1; continue
        master.grad = grad; opt.step(); opt.zero_grad(set_to_none=True)
        w.data.copy_(master.detach().to(w.dtype))
    m.eval(); w.requires_grad_(False)
    return loss.item(), skipped

if BONUS:
    seed_all()
    big = json.loads(train_bpe(16_000).to_str())["model"]
    pc = adjacent_pairs(new_tok)
    print(f"pre-tokens of the train corpus that the pruned tokenizer splits: {sum(pc.values()):,} adjacent token pairs")
    b1, FT_STEPS = [], 100 if DEVICE == "cuda" else 0
    for n_new in [0, 100, 500]:
        nj, new = extend(new_json, n_new, big, pc)
        nt = tokenizer_from_json(nj)
        m = load_model(); prune_model_embeddings(m, old2new); remap_special_ids(m, old2new)
        emb = m.get_input_embeddings().weight.data
        size = max(nj["model"]["vocab"].values()) + 1
        padded = int(math.ceil(max(size, emb.shape[0]) / 64) * 64)
        w = emb.new_zeros(padded, emb.shape[1]); w[:emb.shape[0]] = emb
        for t, a, b in new:
            w[nj["model"]["vocab"][t]] = emb[[x.id for x in new_tok.model.tokenize(t)]].float().mean(0).to(emb.dtype)
        m.set_input_embeddings(torch.nn.Embedding(padded, emb.shape[1])); m.get_input_embeddings().weight.data = w
        m.get_output_embeddings().weight = m.get_input_embeddings().weight; m.config.vocab_size = padded
        _, nt_tok, nt_bytes = token_usage(nt, train_docs)
        row = {"new tokens": len(new), "examples": ", ".join(show(t) for t, _, _ in new[:8]),
               "fertility (train)": nt_tok / n_words, "tokens/byte (held)": None,
               "bits/byte (held)": None, "bits/byte after emb. fine-tune": None}
        r = bpb(m, nt, held_docs, eos_new); row["tokens/byte (held)"] = r["tokens_per_byte"]; row["bits/byte (held)"] = r["bits_per_byte"]
        if FT_STEPS:
            finetune_embeddings(m, nt, eos_new, FT_STEPS)
            row["bits/byte after emb. fine-tune"] = bpb(m, nt, held_docs, eos_new)["bits_per_byte"]
        b1.append(row); print(row)
        del m; torch.cuda.empty_cache()
    tb1 = pd.DataFrame(b1).set_index("new tokens")
    display(tb1.style.format({c: "{:.4f}" for c in tb1.columns if c != "examples"}))
    RESULTS["B1"] = b1
    lap("B1")

# %% [markdown]
# Little room is left for extension: every list word is already a single token, digits are split one by one by the
# pre-tokenizer, and only rare names (`␣Assam`, `␣Gowa`, …) are split. Fertility improves by well under 1%, and the
# mean-initialised new tokens cost a little bits per byte (they compete with the old way of spelling the same string)
# until they are trained. The embedding-only fine-tune helps every variant by about the same amount: it teaches the
# model the caveman *style*, which matters far more than a few more tokens.

# %% [markdown]
# ## Bonus B4 — serving: decode throughput and peak memory

# %%
@torch.no_grad()
def serve(model_dir, batch, new_tokens=128, reps=2):
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=DTYPE).to(DEVICE).eval()
    f = PreTrainedTokenizerFast.from_pretrained(model_dir); f.padding_side = "left"
    x = f([PROMPTS[i % len(PROMPTS)] for i in range(batch)], return_tensors="pt", padding=True).to(DEVICE)
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=f.pad_token_id)
    m.generate(**x, max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=f.pad_token_id)   # warm-up
    if DEVICE == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        m.generate(**x, **kw)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else float("nan")
    del m; torch.cuda.empty_cache()
    return {"tokens/s": batch * new_tokens / dt, "ms/step": dt / new_tokens * 1e3, "peak MiB": peak}

if BONUS:
    b4 = []
    for d, label in dict.fromkeys([("ckpt_original", "original"), (out_dir, f"pruned m={M}"), (ship_dir, f"pruned m={SHIP_M}")]):
        for b in [1, 8]:
            b4.append({"model": label, "batch": b, **serve(d, b)}); print(b4[-1])
    tb4 = pd.DataFrame(b4).set_index(["model", "batch"])
    display(tb4.style.format("{:.1f}"))
    RESULTS["B4"] = b4
    lap("B4")

# %% [markdown]
# The pruned model needs ~25% less GPU memory (the 151,936 × 896 embedding is 27% of the weights). Decode throughput
# barely changes on a GPU: generating with a 0.5B model is dominated by 24 transformer layers and kernel-launch overhead,
# not by the output head. On a CPU or a phone, where the head's matrix-vector product is a real share of the time, the gain is larger.

# %% [markdown]
# ## Summary and reproducibility

# %%
RESULTS["times"] = TIMES
RESULTS["env"] = {"device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", "torch": torch.__version__,
                  "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "tiktoken": tiktoken.__version__,
                  "seed": SEED, "dtype": str(DTYPE)}
json.dump(RESULTS, open("results.json", "w"), indent=1, default=str)
print(json.dumps(TIMES), f"| total runtime {time.time() - T_START:.0f} s on {RESULTS['env']['device']}")
