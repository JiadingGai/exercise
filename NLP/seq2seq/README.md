# Seq2Seq Transformer MT Exercise

This directory trains a small German-to-English Transformer model on the
`bentrevett/multi30k` dataset mirror from Hugging Face.

## Environment Used

The sanity runs in this directory used:

```text
Python 3.12.13
torch 2.12.0
spacy 3.8.14
de_core_news_sm 3.8.0
en_core_web_sm 3.8.0
requests 2.34.2
numpy 2.4.6
tqdm 4.68.2
```

`torchtext 0.18.0` was installed in the local test venv, but it is intentionally
not used by `mytransformers.py`: it failed to import against `torch 2.12.0` on
this machine.

## Install

From this directory:

```bash
/opt/homebrew/bin/python3.12 -m venv venv_latest_torch
./venv_latest_torch/bin/python -m pip install --upgrade pip setuptools wheel
./venv_latest_torch/bin/python -m pip install -r requirements.txt
./venv_latest_torch/bin/python -m spacy download de_core_news_sm
./venv_latest_torch/bin/python -m spacy download en_core_web_sm
```

The repo vendors a small JSONL snapshot under `data/multi30k`, and
`mytransformers.py` uses this local copy by default for train/validation/test
data. If those files are missing, the script downloads them from
`bentrevett/multi30k`; set `MULTI30K_CACHE_DIR` to use another local path.

## Dataset Snapshot

The vendored files are mirrored from:

```text
https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/train.jsonl
https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/val.jsonl
https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/test.jsonl
```

Snapshot sizes and SHA-256 checksums:

```text
train.jsonl  29,000 lines  de8a1bb324fcf14c44b17b2baee60304c9aba9e5f5889da8a955ffffebc71abb
val.jsonl     1,014 lines  f23e04140ebb001299365cd1d041da54c6cdd970b6eba64b7124242580f05e52
test.jsonl    1,000 lines  0b997c04e05614fdca57c7997bb42822db69b4177d6dc96d80a5a6af8a758588
```

## Run

One-epoch smoke test:

```bash
PYTHONUNBUFFERED=1 \
N_EPOCHS=1 \
BATCH_SIZE=128 \
BLEU_EVAL_LIMIT=100 \
SAMPLE_TRANSLATION_EVERY=1 \
LOSS_LOG_EVERY=50 \
./venv_latest_torch/bin/python mytransformers.py
```

Ten-epoch sanity run:

```bash
PYTHONUNBUFFERED=1 \
N_EPOCHS=10 \
BATCH_SIZE=128 \
BLEU_EVAL_LIMIT=100 \
SAMPLE_TRANSLATION_EVERY=1 \
LOSS_LOG_EVERY=50 \
./venv_latest_torch/bin/python mytransformers.py
```

Useful environment variables:

```text
MULTI30K_CACHE_DIR         Dataset JSONL cache directory.
N_EPOCHS                  Number of training epochs. Default: 30.
BATCH_SIZE                Training batch size. Default: 128.
BLEU_EVAL_LIMIT           Number of validation/test examples for BLEU. Default: 100.
SAMPLE_TRANSLATION_EVERY  Print greedy sample translation every N epochs. Default: 1.
LOSS_LOG_EVERY            Print batch loss every N steps. 0 disables it. Default: 50.
```

## Notes

`mytransformers.py` uses `calculate_bleu` plus a local, unsmoothed
`corpus_bleu` implementation for lightweight training feedback. It is useful
for sanity checks but is not a standardized sacreBLEU score for benchmark
reporting.

For each dataset example $(x_i, y_i)$, `calculate_bleu` greedily decodes
$x_i$, removes one trailing `<eos>` token if present, tokenizes the gold English
target $y_i$, and then calls `corpus_bleu` with one reference per candidate.
Let $\hat{C}_i$ be that decoded candidate and let $R_i$ be its one-reference
set:

$$
D = \{(x_i, y_i)\}_{i=1}^{m},
\qquad
C = \{\hat{C}_i\}_{i=1}^{m},
\qquad
R = \{R_i\}_{i=1}^{m}
$$

$$
B_D = B(C, R)
$$

Here, $B(C, R)$ denotes the local `corpus_bleu` calculation.

For candidates $C$, references $R$, $N=4$, and uniform weights
$w_n = \frac{1}{4}$, `corpus_bleu` computes modified n-gram precision with
clipped counts. Here, $G_n(s)$ is the multiset of n-grams in sequence $s$, and
$N(g, s)$ is the count of n-gram $g$ in $s$:

$$
p_n =
\frac{
  \sum_i \sum_{g \in G_n(\hat{C}_i)}
    \min\left(N(g, \hat{C}_i), \max_{r \in R_i} N(g, r)\right)
}{
  \sum_i \max\left(|\hat{C}_i| - n + 1, 0\right)
}
$$

It then applies the standard corpus brevity penalty using the reference length
$r$ closest to each candidate length and total candidate length $c$:

$$
BP = \exp\left(\min\left(1 - \frac{r}{c}, 0\right)\right),
\qquad
c = \sum_i |\hat{C}_i|
$$

$$
BLEU =
BP \cdot \exp\left(\frac{1}{4}\sum_{n=1}^{4}\log p_n\right)
$$

Because this implementation is unsmoothed, it returns `0.0` when $c = 0$ or
when any clipped n-gram count is zero.
