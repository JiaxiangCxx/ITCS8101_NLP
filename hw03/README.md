# ITCS 8101 NLP — HW03: Feed-Forward Neural Language Model

Homework 3 for ITCS 8101 (Natural Language Processing), UNC Charlotte, Spring 2026.

A fully connected neural language model (FFLM) in PyTorch. The model takes the
pre-trained GloVe embeddings of the previous **K** words, concatenates them, passes
them through **H** hidden layers of **N** ReLU units, and predicts the next word with
a softmax over a 10K-word vocabulary.

```
K*EDim -> [Linear + ReLU] x H -> Linear -> Softmax
```

## Repository layout

```
code/
  fflm.py                         model, training, evaluation, interactive + generation modes
  lm_model.pt                     trained checkpoint (its K/H/N are stored in the file)
  traces_k4n100.txt               training trace, baseline (K=4, H=2, N=100)
  traces_greedy_true.txt          prompt mode, greedy decoding
  traces_greedy_false.txt         prompt mode, sampling
  Top10prep.txt                   interactive mode: top-10 next-word predictions
  traces_bonus_*.txt              bonus experiments (see Results)
data/
  vocab.embeddings.txt            10K training vocabulary + its GloVe-100d vectors
  tiny/                           TinyStories-style corpus (train 4M / valid 200K / test 4M tokens)
  wiki/                           WikiText corpus (alternative)
```

GloVe is **not** included (≈2 GB). It is only needed for `-train`; the other modes
use `data/vocab.embeddings.txt`. To retrain, download
[glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) and unzip it into
`data/glove.6B/`.

## Usage

Run from `code/` (default paths are relative to it):

```bash
pip install -r ../requirements.txt

python fflm.py -train                       # train (default K=4, H=2, N=100, 1 epoch)
python fflm.py -use                         # type K words, see top-10 next words
python fflm.py -prompt -greedy True         # autoregressive generation (greedy)
python fflm.py -prompt -greedy False        # autoregressive generation (sampling)
python fflm.py -evaluate                    # perplexity on the test corpus
```

Hyperparameters: `-K`, `-H`, `-N`, `-E` (epochs), `-B` (batch size), `-VSize`.

## Results

Validation perplexity after 1 epoch on the tiny corpus (GloVe-100d inputs):

| Configuration | Valid PPL |
|---|---|
| K=4, H=2, N=100 (baseline) | 34.01 |
| K=4, H=2, N=256 | **29.21** |
| K=6, H=2, N=100 | 42.43 |
| K=4, H=2, N=100 + dropout 0.2 | 34.78 |

Example greedy generation:

```
Prompt:    the bird then flew
Generated: the bird then flew down and saw a big tree .
```
