"""
Feed-Forward Neural Network Language Model (FFLM)
==================================================
A fully connected NN trained for language modeling using pre-trained GloVe embeddings.
The model takes K consecutive word embeddings as input and predicts the next word.

Usage:
    python fflm.py -train
    python fflm.py -use
    python fflm.py -evaluate
    python fflm.py -prompt
"""

import argparse
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Model Definition
# ============================================================================

class FFLM(nn.Module):
    """Feed-forward neural network language model.

    Architecture: K*EDim -> [Linear+ReLU]*H -> Linear -> Softmax
    Takes concatenated embeddings of K words, passes through H hidden layers
    of N neurons each (ReLU activation), and outputs a probability distribution
    over the vocabulary.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden):
        super().__init__()
        layers = []

        # YOUR CODE HERE: BEGIN
        # First hidden layer: input -> hidden.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout(p=0.2))

        # Additional hidden layers: hidden -> hidden.
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
    
        # Output layer: hidden -> vocab size (no activation; handled by CrossEntropyLoss).
        layers.append(nn.Linear(hidden_dim, output_dim))

       # YOUR CODE HERE: END

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Vocabulary & Embeddings
# ============================================================================

def load_embedding_vocab(embeddings_file):
    """Read the embeddings file and return the set of all tokens."""
    vocab_emb = set()
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.split(' ', 1)[0]
            vocab_emb.add(token)
    print(f"  Loaded embedding vocabulary: {len(vocab_emb)} tokens.")
    return vocab_emb


def build_train_vocab(corpus_file, vocab_emb, vsize):
    """Build a frequency-based vocabulary from the training corpus.

    Tokens are lowercased. Only tokens present in vocab_emb are kept.
    The vocabulary is pruned to the top 'vsize' most frequent tokens.
    Returns a dictionary mapping token -> frequency.
    """
    freq = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        text = f.read()
    for token in text.split():
        token = token.lower()
        if token in vocab_emb:
            freq[token] = freq.get(token, 0) + 1
    # Keep only the top 'vsize' most frequent tokens.
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab_train = {t: c for t, c in sorted_tokens[:vsize]}
    print(f"  Training vocabulary: {len(vocab_train)} tokens (from {len(freq)} unique).")
    return vocab_train


def load_embeddings_for_vocab(embeddings_file, vocab_train):
    """Load embeddings only for tokens in vocab_train.

    Returns a dict mapping token -> list of floats (embedding vector).
    """
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            token = parts[0]
            if token in vocab_train:
                embeddings[token] = [float(x) for x in parts[1:]]
    print(f"  Loaded embeddings for {len(embeddings)} tokens.")
    return embeddings


def build_vocab_and_embeddings(embeddings_file, corpus_file, vsize):
    """Build vocabulary and embedding tensor from embeddings file and corpus.

    Returns:
        token2idx: dict mapping token string -> row index in tembeddings.
        tembeddings: 2D tensor of shape (vsize+1, EDim), last row is <unk>.
        EDim: embedding dimensionality.
    """
    print("Building vocabulary and embeddings...")
    vocab_emb = load_embedding_vocab(embeddings_file)
    vocab_train = build_train_vocab(corpus_file, vocab_emb, vsize)
    emb_dict = load_embeddings_for_vocab(embeddings_file, vocab_train)

    # Determine embedding dimension from the first entry.
    EDim = len(next(iter(emb_dict.values())))

    # Build token2idx and tembeddings.
    token2idx = {}
    emb_list = []
    for idx, (token, vec) in enumerate(emb_dict.items()):
        token2idx[token] = idx
        emb_list.append(vec)

    tembeddings = torch.tensor(emb_list, dtype=torch.float32)

    # Create <unk> embedding as the mean of all token embeddings.
    unk_emb = tembeddings.mean(dim=0, keepdim=True)
    tembeddings = torch.cat([tembeddings, unk_emb], dim=0)
    token2idx['<unk>'] = tembeddings.size(0) - 1

    print(f"  Final vocabulary: {len(token2idx)} tokens, EDim = {EDim}.")
    return token2idx, tembeddings, EDim


def load_vocab_and_embeddings(vocab_file):
    """Load token2idx and tembeddings from a saved vocab.embeddings.txt file.

    Returns:
        token2idx: dict mapping token -> row index.
        tembeddings: 2D tensor of shape (num_tokens, EDim).
        EDim: embedding dimensionality.
    """
    print(f"Loading vocabulary from {vocab_file}...")
    tokens = []
    emb_list = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            tokens.append(parts[0])
            emb_list.append([float(x) for x in parts[1:]])
    token2idx = {t: i for i, t in enumerate(tokens)}
    tembeddings = torch.tensor(emb_list, dtype=torch.float32)
    EDim = tembeddings.size(1)
    print(f"  Loaded {len(token2idx)} tokens, EDim = {EDim}.")
    return token2idx, tembeddings, EDim


# ============================================================================
# Data Preparation
# ============================================================================

def tokenize_corpus(corpus_file, token2idx):
    """Read a corpus file and convert to a list of token indices.

    Tokens are lowercased; out-of-vocabulary tokens map to <unk>.
    Returns a list of integer indices.
    """
    unk_idx = token2idx['<unk>']
    with open(corpus_file, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.split()
    indices = [token2idx.get(t.lower(), unk_idx) for t in tokens]
    return indices


def create_examples(token_indices, K):
    """Create input-output pairs for language modeling.

    For each position i >= K, the input is the indices of tokens [i-K, ..., i-1]
    and the label is the index of token i.
    Returns:
        inputs: LongTensor of shape (num_examples, K)
        labels: LongTensor of shape (num_examples,)
    """
    n = len(token_indices)
    if n <= K:
        return torch.zeros(0, K, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    indices = torch.tensor(token_indices, dtype=torch.long)

    # YOUR CODE HERE: BEGIN
    inputs = indices.unfold(dimension=0, size=K, step=1)  # (n-K+1, K)
    labels = indices[K:] # (n-K,)
    inputs = inputs[:-1] # (n-K, K)
    
    # YOUR CODE HERE END
     
    return inputs, labels


def lookup_embeddings(input_indices, tembeddings):
    """Look up and concatenate embeddings for input token indices.

    Args:
        input_indices: LongTensor of shape (batch, K).
        tembeddings: 2D tensor of shape (vocab_size+1, EDim).
    Returns:
        Tensor of shape (batch, K * EDim) with concatenated embeddings.
    """
    embs = tembeddings[input_indices]             # (batch, K, EDim)
    return embs.view(embs.size(0), -1)            # (batch, K * EDim)


# ============================================================================
# Compute Loss (vectorized)
# ============================================================================

def compute_loss(model, inputs, labels, tembeddings, criterion, batch_size=2048):
    """Compute average cross-entropy loss over all examples in a vectorized manner."""
    model.eval()

    total_loss = 0.0
    n = inputs.size(0)
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            # YOUR CODE HERE: BEGIN
            end = min(start + batch_size, n)
            x_ids = inputs[start:end]  # (batch, K)
            y = labels[start:end]      # (batch,)
            x = lookup_embeddings(x_ids, tembeddings)  # (batch, K*EDim)
            logits = model(x)          # (batch, vocab_size)
            loss = criterion(logits, y)
            total_loss += loss.item() * (end - start)

            # YOUR CODE HERE: END

    model.train()
    
    return total_loss / n


# ============================================================================
# Save / Load Utilities
# ============================================================================

def load_model(model_path, vocab_size, EDim):
    """Load a saved model checkpoint and reconstruct the FFLM.

    The checkpoint contains model weights and architecture parameters
    (K, H, N, VSize, EDim), so these do not need to be specified separately.

    Args:
        model_path: path to the saved .pt checkpoint file.
        vocab_size: number of tokens (including <unk>), from loaded vocabulary.
        EDim: embedding dimensionality, from loaded vocabulary.
    Returns:
        model: the reconstructed FFLM in eval mode.
        params: dict with saved architecture parameters (K, H, N, VSize, EDim).
    """
    checkpoint = torch.load(model_path, weights_only=True)
    params = {k: checkpoint[k] for k in ('K', 'H', 'N', 'VSize', 'EDim')}
    input_dim = params['K'] * EDim
    model = FFLM(input_dim, params['N'], vocab_size, params['H'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from {model_path} "
          f"(K={params['K']}, H={params['H']}, N={params['N']}).")
    return model, params


def save_vocab_embeddings(token2idx, tembeddings, filepath):
    """Save token strings and their embeddings in GloVe-style text format."""
    # Invert token2idx to get idx2token.
    idx2token = {i: t for t, i in token2idx.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx in range(tembeddings.size(0)):
            token = idx2token[idx]
            vec = tembeddings[idx].tolist()
            vec_str = ' '.join(f'{v:.6f}' for v in vec)
            f.write(f'{token} {vec_str}\n')
    print(f"  Saved vocabulary embeddings to {filepath}.")


# ============================================================================
# train()
# ============================================================================

def train(args):
    """Train the feed-forward language model."""
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    # 1. Build vocabulary and embeddings from the training corpus.
    token2idx, tembeddings, EDim = build_vocab_and_embeddings(
        args.embeddings_file, args.train_corpus, args.VSize
    )
    vocab_size = len(token2idx)  # VSize + 1 (including <unk>)

    # 2. Tokenize training and validation corpora.
    print("Tokenizing training corpus...")
    train_indices = tokenize_corpus(args.train_corpus, token2idx)
    print(f"  Training tokens: {len(train_indices)}")

    print("Tokenizing validation corpus...")
    valid_indices = tokenize_corpus(args.valid_corpus, token2idx)
    print(f"  Validation tokens: {len(valid_indices)}")

    # 3. Create training and validation examples.
    print("Creating training examples...")
    train_inputs, train_labels = create_examples(train_indices, args.K)
    print(f"  Training examples: {train_inputs.size(0)}")

    print("Creating validation examples...")
    valid_inputs, valid_labels = create_examples(valid_indices, args.K)
    print(f"  Validation examples: {valid_inputs.size(0)}")

    # 4. Create the model.
    input_dim = args.K * EDim
    model = FFLM(input_dim, args.N, vocab_size, args.H)
    print(f"\nModel architecture:\n{model}\n")

    # 5. Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop.
    num_examples = train_inputs.size(0)
    B = args.B
    batch_count = 0

    for epoch in range(1, args.E + 1):
        print(f"--- Epoch {epoch}/{args.E} ---")
        # Shuffle training data.
        perm = torch.randperm(num_examples)
        train_inputs = train_inputs[perm]
        train_labels = train_labels[perm]

        epoch_loss = 0.0
        epoch_batches = 0

        for start in range(0, num_examples, B):
            end = min(start + B, num_examples)
            x = lookup_embeddings(train_inputs[start:end], tembeddings)
            y = train_labels[start:end]

            # YOUR CODE HERE: BEGIN
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            # YOUR CODE HERE: END

            epoch_loss += loss.item()
            epoch_batches += 1
            batch_count += 1

            # Print validation loss every 100 minibatches.
            if batch_count % 100 == 0:
                val_loss = compute_loss(
                    model, valid_inputs, valid_labels, tembeddings, criterion
                )
                avg_train = epoch_loss / epoch_batches
                print(f"  Batch {batch_count:>6d} | "
                      f"Train loss (epoch avg): {avg_train:.4f} | "
                      f"Valid loss: {val_loss:.4f} | "
                      f"Valid PPL: {math.exp(val_loss):.2f}")

        # End-of-epoch stats.
        avg_epoch_loss = epoch_loss / epoch_batches
        val_loss = compute_loss(
            model, valid_inputs, valid_labels, tembeddings, criterion
        )
        print(f"  Epoch {epoch} done | "
              f"Avg train loss: {avg_epoch_loss:.4f} | "
              f"Valid loss: {val_loss:.4f} | "
              f"Valid PPL: {math.exp(val_loss):.2f}\n")

    # 7. Save model checkpoint with architecture parameters.
    model_path = 'lm_model.pt'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'K': args.K,
        'H': args.H,
        'N': args.N,
        'VSize': args.VSize,
        'EDim': EDim,
    }
    torch.save(checkpoint, model_path)
    print(f"  Model saved to {model_path} (K={args.K}, H={args.H}, N={args.N}).")

    vocab_emb_path = '../data/vocab.embeddings.txt'
    save_vocab_embeddings(token2idx, tembeddings, vocab_emb_path)

    print("Training complete.")


# ============================================================================
# use()
# ============================================================================

def use(args):
    """Interactive mode: input K tokens, see top-10 predicted next tokens."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)

    # Load vocabulary and model (K, H, N read from checkpoint).
    vocab_file = '../data/vocab.embeddings.txt'
    token2idx, tembeddings, EDim = load_vocab_and_embeddings(vocab_file)
    vocab_size = len(token2idx)
    unk_idx = token2idx['<unk>']

    model, params = load_model('lm_model.pt', vocab_size, EDim)
    K = params['K']

    # Invert token2idx for decoding.
    idx2token = {i: t for t, i in token2idx.items()}

    print(f"\nEnter {K} space-separated tokens (empty to quit):\n")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            print("Exiting interactive mode.")
            break

        tokens = user_input.lower().split()

        # Pad with <unk> if fewer than K tokens; truncate to last K.
        if len(tokens) < K:
            tokens = ['<unk>'] * (K - len(tokens)) + tokens
        elif len(tokens) > K:
            tokens = tokens[-K:]

        # Map to indices, replacing OOV with <unk>.
        indices = [token2idx.get(t, unk_idx) for t in tokens]
        processed = [idx2token[i] for i in indices]
        print(f"  Processed tokens: {' '.join(processed)}")

        # Predict, excluding <unk> from the output distribution.

        input_tensor = torch.tensor([indices], dtype=torch.long)  # (1, K)
        x = lookup_embeddings(input_tensor, tembeddings)          # (1, K*EDim)
        with torch.no_grad():
            # YOUR CODE HERE: BEGIN
            # probs = []
            logits = model(x)                         # (1, V)
            logits[0, unk_idx] = -1e9                 # suppress <unk>
            probs = torch.softmax(logits, dim=1) 
            # YOUR CODE HERE: END

        # Top 10 predictions.
        top_probs, top_indices = torch.topk(probs, 10, dim=1)
        print("  Top 10 predictions:")
        for i in range(10):
            tok = idx2token[top_indices[0, i].item()]
            prob = top_probs[0, i].item()
            print(f"    {tok:<20s} {prob:.6f}")
        print()


# ============================================================================
# prompt()
# ============================================================================

def generate_next_token(model, context_indices, tembeddings, greedy, unk_idx):
    """Generate the next token index given a context window of K token indices.

    The <unk> token is excluded from generation.

    Args:
        model: the FFLM model (in eval mode).
        context_indices: list of K integer indices (the current context window).
        tembeddings: embedding tensor.
        greedy: if True, pick argmax; if False, sample from the distribution.
        unk_idx: index of the <unk> token to suppress from generation.
    Returns:
        The integer index of the generated token.
    """
    input_tensor = torch.tensor([context_indices], dtype=torch.long)  # (1, K)
    x = lookup_embeddings(input_tensor, tembeddings)                  # (1, K*EDim)
    with torch.no_grad():
        # YOUR CODE HERE: BEGIN
        index = -1

        logits = model(x)       # (1, V)
        logits[0, unk_idx] = -1e9  # suppress <unk>
        probs = torch.softmax(logits, dim=1)

        if greedy:
            index = torch.argmax(probs, dim=1).item()
        else:
            index = torch.multinomial(probs, num_samples=1).item()

        return index
        # YOUR CODE HERE: END


def prompt(args, greedy=True):
    """Autoregressive text generation: input tokens, generate up to 15 more.

    Loads the saved model and vocabulary, then interactively reads a token
    sequence from the user. The model generates tokens one at a time,
    conditioning on the last K tokens, and stops when a sentence-ending
    punctuation token ('.', '!', '?') is produced or 15 tokens are generated.

    Args:
        args: parsed command line arguments.
        greedy: if True (default), always pick the most probable next token;
                if False, sample from the predicted distribution.
    """
    print("=" * 60)
    print("PROMPT MODE" + (" (greedy)" if greedy else " (sampling)"))
    print("=" * 60)

    STOP_TOKENS = {'.', '!', '?'}
    MAX_GEN = 15

    # Load vocabulary and model (K, H, N read from checkpoint).
    vocab_file = '../data/vocab.embeddings.txt'
    token2idx, tembeddings, EDim = load_vocab_and_embeddings(vocab_file)
    vocab_size = len(token2idx)
    unk_idx = token2idx['<unk>']

    model, params = load_model('lm_model.pt', vocab_size, EDim)
    K = params['K']

    # Invert token2idx for decoding.
    idx2token = {i: t for t, i in token2idx.items()}

    print(f"\nEnter a sequence of tokens to use as prompt (empty to quit):\n")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            print("Exiting prompt mode.")
            break

        tokens = user_input.lower().split()

        # Pad with <unk> if fewer than K tokens; keep all if more than K.
        if len(tokens) < K:
            tokens = ['<unk>'] * (K - len(tokens)) + tokens

        # Map to indices, replacing OOV with <unk>.
        indices = [token2idx.get(t, unk_idx) for t in tokens]
        processed = [idx2token[i] for i in indices]
        print(f"  Prompt: {' '.join(processed)}")

        # Autoregressive generation: use the last K indices as context.
        generated = []
        context = list(indices[-K:])

        for _ in range(MAX_GEN):
            next_idx = generate_next_token(model, context, tembeddings, greedy, unk_idx)
            next_token = idx2token[next_idx]
            generated.append(next_token)

            # Stop if a sentence-ending punctuation token is generated.
            if next_token in STOP_TOKENS:
                break

            # Slide the context window: drop oldest, append new token.
            context = context[1:] + [next_idx]

        print(f"  Generated: {' '.join(processed)} {' '.join(generated)}\n")


# ============================================================================
# evaluate()
# ============================================================================

def evaluate(args):
    """Evaluate the trained model on the test corpus and report perplexity."""
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Load vocabulary and model (K, H, N read from checkpoint).
    vocab_file = '../data/vocab.embeddings.txt'
    token2idx, tembeddings, EDim = load_vocab_and_embeddings(vocab_file)
    vocab_size = len(token2idx)

    model, params = load_model('lm_model.pt', vocab_size, EDim)
    K = params['K']

    # Tokenize test corpus and create examples.
    print(f"Tokenizing test corpus: {args.test_corpus}")
    test_indices = tokenize_corpus(args.test_corpus, token2idx)
    print(f"  Test tokens: {len(test_indices)}")

    test_inputs, test_labels = create_examples(test_indices, K)
    print(f"  Test examples: {test_inputs.size(0)}")

    # YOUR CODE HERE: BEGIN
    avg_loss = 0
    perplexity = 0


    
    # YOUR CODE HERE: END
    crition = nn.CrossEntropyLoss()
    avg_loss = compute_loss(model, test_inputs, test_labels, tembeddings, crition)
    perplexity = math.exp(avg_loss)
    print(f"\n  Test Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"  Test Perplexity:         {perplexity:.2f}\n")


# ============================================================================
# main()
# ============================================================================

def main():
    """Parse command line arguments and dispatch to train/use/evaluate/prompt."""
    parser = argparse.ArgumentParser(
        description='Feed-Forward Neural Network Language Model (FFLM)')

    # Mode of operation (mutually exclusive).
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-train', action='store_true', help='Train the model.')
    mode.add_argument('-use', action='store_true', help='Interactive mode.')
    mode.add_argument('-evaluate', action='store_true', help='Evaluate on test corpus.')
    mode.add_argument('-prompt', action='store_true', help='Autoregressive generation.')

    # Model hyperparameters.
    parser.add_argument('-H', type=int, default=2,
                        help='Number of hidden layers (default: 2).')
    parser.add_argument('-N', type=int, default=100,
                        help='Number of neurons per hidden layer (default: 100).')
    parser.add_argument('-K', type=int, default=4,
                        help='Number of input context words (default: 4).')
    parser.add_argument('-E', type=int, default=1,
                        help='Number of training epochs (default: 1).')
    parser.add_argument('-B', type=int, default=100,
                        help='Minibatch size (default: 100).')
    parser.add_argument('-VSize', type=int, default=10000,
                        help='Vocabulary size (default: 10000).')

    # File paths.
    parser.add_argument('-embeddings_file', type=str,
                        default='../data/glove.6B/glove.6B.100d.txt',
                        help='Path to GloVe embeddings file.')
    parser.add_argument('-train_corpus', type=str,
                        default='../data/tiny/tiny.train.tokens.4M', # default='../data/wiki.train.tokens',
                        help='Path to tokenized training corpus.')
    parser.add_argument('-valid_corpus', type=str,
                        default='../data/tiny/tiny.valid.tokens.200K', # default='../data/wiki.valid.tokens',
                        help='Path to tokenized validation corpus.')
    parser.add_argument('-test_corpus', type=str,
                        default='../data/tiny/tiny.test.tokens.4M', # default='../data/wiki.test.tokens',
                        help='Path to tokenized test corpus.')

    # Generation parameters.
    parser.add_argument('-greedy', type=str, default='True',
                        choices=['True', 'False'],
                        help='Greedy decoding in prompt mode (default: True).')

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.use:
        use(args)
    elif args.evaluate:
        evaluate(args)
    elif args.prompt:
        prompt(args, greedy=(args.greedy == 'True'))


if __name__ == '__main__':
    main()
