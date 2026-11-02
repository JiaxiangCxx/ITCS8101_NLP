# ITCS 8101 Natural Language Processing: Coursework

My homework for ITCS 8101 (Natural Language Processing), UNC Charlotte, Spring 2026.
Each folder is self-contained and has its own README with setup, usage and results.

| Folder | Topic | Highlights |
|---|---|---|
| [`hw03/`](hw03) | Feed-forward neural language model | GloVe-100d inputs, K-word context, PyTorch; valid PPL 29.2 (best) |
| [`hw05/`](hw05) | Transformer decoder language model | written from scratch: multi-head attention, sinusoidal PE, weight tying, Pre-/Post-LN; valid PPL 11.0; plus a parameter-count derivation |
| [`hw07/`](hw07) | Restaurant reservation chatbot | MCP server (FastMCP) and a Gemini function-calling dialog manager with dialog-state tracking |
| [`AI/`](AI) | Search and CSP exercises | N-Queens hill climbing and map coloring with backtracking, forward checking and propagation |

## Not included

Some large or third-party files are left out to keep the repository small:

- `hw03/data/glove.6B/`: GloVe embeddings (≈2 GB), only needed for training.
  Download them from <https://nlp.stanford.edu/projects/glove/>.
- `hw07/chatbot_server/data/restaurants.json`: a 140 MB sample of the Yelp
  Academic Dataset, which may not be redistributed.
- `hw07/chatbot_server/data/embeddings/`: precomputed vector-DB files (≈790 MB).

## Requirements

Python 3.10+. Each homework lists its own dependencies in `requirements.txt`:
PyTorch for hw03/hw05, and `fastmcp` plus `google-genai` for hw07.
