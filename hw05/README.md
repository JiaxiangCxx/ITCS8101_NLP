# ITCS 8101 NLP — HW05: Transformer Decoder Language Model

Homework 5 for ITCS 8101 (Natural Language Processing), UNC Charlotte, Spring 2026.

Two parts:

1. **Theory** (`hw05-theory.ipynb` / `.pdf`): parameter counts for a Transformer,
   covering embeddings, multi-head attention, the FFN and LayerNorm, as functions of
   L, h, d, d_ff and |V|.
2. **Code** (`code/transformerlm.py`): a multi-layer Transformer decoder LM written
   from scratch in PyTorch:
   - token embeddings learned from scratch, sinusoidal positional encodings
   - masked multi-head self-attention and a position-wise FFN (ReLU or GELU)
   - Post-LN by default, Pre-LN with `-norm_first`
   - output layer weight-tied to the input embedding
   - trained per story with `<bos>` / `<eos>` and padded minibatches

## Repository layout

```
hw05-theory.ipynb / hw05-theory.pdf   theory answers
code/
  transformerlm.py                    model, training, evaluation, interactive + generation modes
  lm_model.pt                         trained checkpoint (its architecture is stored in the file)
  traces.txt                          traces for train / use / prompt / evaluate
  bonus.txt                           bonus experiments (see Results)
data/
  vocab.txt                           10K vocabulary (+ <unk>, <bos>, <eos>)
  tiny/                               TinyStories-style corpus (train 4M / valid 200K / test 4M tokens)
```

## Usage

Run from `code/` (default paths are relative to it). Uses CUDA if available.

```bash
pip install -r ../requirements.txt

python transformerlm.py -train                    # default: 8 layers, d_model 128, 4 heads, FFN 512, 4 epochs
python transformerlm.py -train -norm_first        # Pre-LN variant
python transformerlm.py -use                      # interactive next-word prediction
python transformerlm.py -prompt -greedy True      # greedy generation
python transformerlm.py -prompt -greedy False     # sampling
python transformerlm.py -evaluate                 # perplexity on the test corpus
```

Other options: `-num_layers`, `-d_model`, `-num_heads`, `-dim_feedforward`,
`-activation {relu,gelu}`, `-dropout`, `-E`, `-B`, `-VSize`, `-lr`.

## Results

Validation perplexity after 4 epochs (from `bonus.txt`):

| Configuration | Valid PPL |
|---|---|
| 8 layers, d_model 128, 4 heads, FFN 512, Post-LN | 11.08 |
| same, Pre-LN (`-norm_first`) | **11.04** |
| 4 layers, d_model 64, 4 heads, FFN 256 | 17.75 |

For comparison, the HW03 feed-forward LM reached a validation perplexity of about 34
on the same corpus.
