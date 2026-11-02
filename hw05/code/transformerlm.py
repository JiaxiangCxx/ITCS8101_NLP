"""
Transformer Decoder Language Model (TransformerLM)
===================================================
A multi-layer Transformer decoder trained for language modeling.
Token embeddings are learned from scratch (no pre-trained embeddings).
Sinusoidal positional encodings are used. The output (softmax) layer
shares weights with the input embedding matrix (weight tying).

Usage:
    python transformerlm.py -train
    python transformerlm.py -use
    python transformerlm.py -evaluate
    python transformerlm.py -prompt
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# Positional Encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Original sine/cosine positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional info added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Multi-Head Self-Attention
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    Splits d_model into num_heads parallel attention heads, each of
    dimension d_k = d_model // num_heads.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for queries, keys, values, and output.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Compute multi-head self-attention.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
            attn_mask: Boolean mask of shape (seq_len, seq_len). True = masked.
            key_padding_mask: Boolean mask of shape (batch, seq_len). True = padded.
        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values: (batch, seq_len, d_model).
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (batch, num_heads, seq_len, d_k).
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention scores: (batch, num_heads, seq_len, seq_len).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (prevent attending to future positions).
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask (prevent attending to padded positions).
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (batch, num_heads, seq_len, d_k).
        context = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, seq_len, d_model).
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection.
        return self.W_o(context)


# ============================================================================
# Position-wise Feed-Forward Network
# ============================================================================

class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network: Linear -> Activation -> Dropout -> Linear."""

    def __init__(self, d_model, dim_feedforward, activation='relu', dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """Args: x of shape (batch, seq_len, d_model). Returns same shape."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ============================================================================
# Transformer Decoder Layer (Block)
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder block: Self-Attention + FFN, each with
    residual connections and layer normalization.

    If norm_first=True (Pre-LN):  x -> LayerNorm -> SubLayer -> residual add
    If norm_first=False (Post-LN): x -> SubLayer -> residual add -> LayerNorm
    """

    def __init__(self, d_model, num_heads, dim_feedforward, activation='relu',
                 dropout=0.1, norm_first=False):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, activation, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Forward pass through one decoder block.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
            attn_mask: Causal mask of shape (seq_len, seq_len).
            key_padding_mask: Padding mask of shape (batch, seq_len).
        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        if self.norm_first:
            # Pre-LN: norm before each sub-layer.
            # YOUR CODE HERE: BEGIN
            # Pre-LN
            # Self-attention
            x_norm = self.norm1(x)
            attn_out = self.self_attn(x_norm, attn_mask, key_padding_mask)
            x = x + self.dropout1(attn_out)

            # Feedforward
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout2(ffn_out)
            # YOUR CODE HERE: END
        else:
            # Post-LN: norm after each sub-layer.
            # YOUR CODE HERE: BEGIN
            # Post-LN
            # Self-attention
            attn_out = self.self_attn(x, attn_mask, key_padding_mask)
            x = self.norm1(x + self.dropout1(attn_out))

            # Feedforward
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))    
            # YOUR CODE HERE: END
        return x


# ============================================================================
# Transformer Language Model
# ============================================================================

class TransformerLM(nn.Module):
    """Multi-layer Transformer decoder language model.

    Architecture:
        Token Embedding (learned) + Sinusoidal Positional Encoding
        -> N x TransformerDecoderLayer
        -> [Optional final LayerNorm if norm_first]
        -> Linear output layer (weights tied to embedding matrix)
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward,
                 activation='relu', dropout=0.1, norm_first=False, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Learned token embeddings, scaled by sqrt(d_model).
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Sinusoidal positional encoding.
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Stack of Transformer decoder layers.
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward,
                                   activation, dropout, norm_first)
            for _ in range(num_layers)
        ])

        # Final layer norm (used when norm_first=True, following standard practice).
        self.norm_first = norm_first
        if norm_first:
            self.final_norm = nn.LayerNorm(d_model)

        # Output projection (weight-tied with embedding matrix).
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight  # Weight tying.

        self._init_weights()

        
    def _init_weights(self):
        """Initialize embedding and linear layer weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Re-zero the padding embedding.
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0)

            
    def _generate_causal_mask(self, seq_len, device):
        """Generate a causal (upper-triangular) mask for self-attention.

        Returns a boolean mask of shape (seq_len, seq_len) where True means
        the position should be masked (cannot attend).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                          diagonal=1)
        return mask

    
    def forward(self, input_ids, key_padding_mask=None):
        """Forward pass through the Transformer LM.

        Args:
            input_ids: LongTensor of shape (batch, seq_len), padded token indices.
            key_padding_mask: Boolean mask of shape (batch, seq_len). True = padded.
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size).
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # YOUR CODE HERE: BEGIN
        # Token embeddings scaled by sqrt(d_model) + positional encoding.
        # Embedding + scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Causal mask: prevent attending to future tokens.
        attn_mask = self._generate_causal_mask(seq_len, device)

        # Pass through each Transformer decoder layer.
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Use final layer norm for Pre-LN variant.
        if self.norm_first:
            x = self.final_norm(x)

        # Project to vocabulary logits (weight-tied with embeddings).
        logits = self.output_layer(x)
        # YOUR CODE HERE: END
        
        return logits


# ============================================================================
# Vocabulary
# ============================================================================

def build_vocab(corpus_file, vsize):
    """Build a frequency-based vocabulary from the training corpus.

    Tokens are lowercased. The vocabulary is pruned to the top 'vsize' most
    frequent tokens. Three special tokens are appended: <unk>, <bos>, <eos>.

    Returns:
        token2idx: dict mapping token string -> integer index.
        vocab_size: total vocabulary size (vsize + 3 special tokens).
    """
    print("Building vocabulary from training corpus...")
    freq = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for token in line.split():
                token = token.lower()
                freq[token] = freq.get(token, 0) + 1

    # Keep only the top VSize most frequent tokens.
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    token2idx = {}
    for idx, (token, _) in enumerate(sorted_tokens[:vsize]):
        token2idx[token] = idx

    # Add special tokens.
    token2idx['<unk>'] = len(token2idx)
    token2idx['<bos>'] = len(token2idx)
    token2idx['<eos>'] = len(token2idx)

    vocab_size = len(token2idx)
    print(f"  Vocabulary: {vocab_size} tokens "
          f"(from {len(freq)} unique, incl. <unk>, <bos>, <eos>).")
    return token2idx, vocab_size


def save_vocab(token2idx, filepath):
    """Save vocabulary to a text file, one token per line, ordered by index."""
    idx2token = {i: t for t, i in token2idx.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx in range(len(idx2token)):
            f.write(f'{idx2token[idx]}\n')
    print(f"  Saved vocabulary to {filepath}.")


def load_vocab(vocab_file):
    """Load vocabulary from a text file (one token per line, index = line number)."""
    print(f"Loading vocabulary from {vocab_file}...")
    token2idx = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.rstrip('\n')
            token2idx[token] = idx
    print(f"  Loaded {len(token2idx)} tokens.")
    return token2idx


# ============================================================================
# Data Preparation
# ============================================================================

def read_stories(corpus_file, token2idx):
    """Read a corpus file and return list of stories as index sequences.

    Each story is one line of text. Blank lines separate stories.
    Each story is prepended with <bos> and appended with <eos>.
    Returns a list of LongTensors, one per story.
    """
    unk_idx = token2idx['<unk>']
    bos_idx = token2idx['<bos>']
    eos_idx = token2idx['<eos>']

    stories = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            indices = [bos_idx] + [token2idx.get(t.lower(), unk_idx) for t in tokens] + [eos_idx]
            stories.append(torch.tensor(indices, dtype=torch.long))
    return stories


def collate_stories(batch):
    """Collate a list of story tensors into padded input/target pairs.

    For each story [bos, w1, w2, ..., wn, eos]:
        input  = [bos, w1, w2, ..., wn]     (predict next token at each step)
        target = [w1, w2, ..., wn, eos]

    Returns:
        inputs: padded LongTensor (batch, max_len), 0-padded.
        targets: padded LongTensor (batch, max_len), -100 for padded positions.
        lengths: LongTensor (batch,), actual sequence lengths.
    """
    input_seqs = [s[:-1] for s in batch]   # drop last token
    target_seqs = [s[1:] for s in batch]   # drop first token
    lengths = torch.tensor([len(s) for s in input_seqs], dtype=torch.long)

    inputs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    targets = pad_sequence(target_seqs, batch_first=True, padding_value=-100)

    return inputs, targets, lengths


# ============================================================================
# Compute Loss
# ============================================================================

def compute_loss(model, stories, criterion, batch_size=64):
    """Compute average cross-entropy loss over all tokens in the stories."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(stories), batch_size):
            batch = stories[start:start + batch_size]
            inputs, targets, lengths = collate_stories(batch)

            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            # Build padding mask: True where padded.
            max_len = inputs.size(1)
            key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

            # YOUR CODE HERE: BEGIN
            logits = model(inputs, key_padding_mask=key_padding_mask)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # YOUR CODE HERE: BEGIN
            
            n_tokens = lengths.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    
    return total_loss / total_tokens


# ============================================================================
# Save / Load Utilities
# ============================================================================

def load_model(model_path, token2idx):
    """Load a saved model checkpoint and reconstruct the TransformerLM.

    Args:
        model_path: path to the saved .pt checkpoint file.
        token2idx: token to index mapping.
    Returns:
        model: the reconstructed TransformerLM in eval mode.
        params: dict with saved architecture parameters.
    """
    checkpoint = torch.load(model_path, weights_only=True)
    params = {k: checkpoint[k] for k in
              ('num_layers', 'd_model', 'num_heads', 'dim_feedforward',
               'activation', 'dropout', 'norm_first', 'VSize')}

    vocab_size = len(token2idx)
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        activation=params['activation'],
        dropout=0.0,  # No dropout at inference.
        norm_first=params['norm_first'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from {model_path} "
          f"(layers={params['num_layers']}, d_model={params['d_model']}, "
          f"heads={params['num_heads']}, ffn={params['dim_feedforward']}, "
          f"act={params['activation']}, norm_first={params['norm_first']}).")
    return model, params


# ============================================================================
# train()
# ============================================================================

def train(args):
    """Train the Transformer language model."""
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Build vocabulary from the training corpus.
    token2idx, vocab_size = build_vocab(args.train_corpus, args.VSize)

    # 2. Read stories from training and validation corpora.
    print("Reading training stories...")
    train_stories = read_stories(args.train_corpus, token2idx)
    print(f"  Training stories: {len(train_stories)}")
    total_train_tokens = sum(len(s) - 1 for s in train_stories)
    print(f"  Training tokens (targets): {total_train_tokens}")

    print("Reading validation stories...")
    valid_stories = read_stories(args.valid_corpus, token2idx)
    print(f"  Validation stories: {len(valid_stories)}")

    # 3. Create the Transformer LM model.
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        activation=args.activation,
        dropout=args.dropout,
        norm_first=args.norm_first,
    ).to(device)
    print(f"\nModel architecture:\n{model}\n")

    # Count trainable parameters.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} total\n")

    # 4. Loss and optimizer.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training loop.
    B = args.B
    num_stories = len(train_stories)
    batch_count = 0

    for epoch in range(1, args.E + 1):
        print(f"--- Epoch {epoch}/{args.E} ---")
        # Shuffle training stories.
        perm = torch.randperm(num_stories).tolist()
        shuffled_stories = [train_stories[i] for i in perm]

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_batches = 0

        for start in range(0, num_stories, B):
            end = min(start + B, num_stories)
            batch = shuffled_stories[start:end]
            inputs, targets, lengths = collate_stories(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            # Build padding mask: True where padded.
            max_len = inputs.size(1)
            key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

            # YOUR CODE HERE: BEGIN
            optimizer.zero_grad()

            logits = model(inputs, key_padding_mask=key_padding_mask)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # YOUR CODE HERE: END

            n_tokens = lengths.sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            epoch_batches += 1
            batch_count += 1

            # Print validation loss every 100 minibatches.
            if batch_count % 100 == 0:
                val_loss = compute_loss(model, valid_stories, criterion)
                avg_train = epoch_loss / epoch_tokens
                print(f"  Batch {batch_count:>6d} | "
                      f"Train loss (epoch avg): {avg_train:.4f} | "
                      f"Valid loss: {val_loss:.4f} | "
                      f"Valid PPL: {math.exp(val_loss):.2f}")

        # End-of-epoch stats.
        avg_epoch_loss = epoch_loss / epoch_tokens
        val_loss = compute_loss(model, valid_stories, criterion)
        print(f"  Epoch {epoch} done | "
              f"Avg train loss: {avg_epoch_loss:.4f} | "
              f"Valid loss: {val_loss:.4f} | "
              f"Valid PPL: {math.exp(val_loss):.2f}\n")

    # 6. Save model checkpoint with architecture parameters.
    model_path = 'lm_model.pt'
    checkpoint = {
        'model_state_dict': model.cpu().state_dict(),
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'dim_feedforward': args.dim_feedforward,
        'activation': args.activation,
        'dropout': args.dropout,
        'norm_first': args.norm_first,
        'VSize': args.VSize,
    }
    torch.save(checkpoint, model_path)
    print(f"  Model saved to {model_path}.")

    vocab_path = '../data/vocab.txt'
    save_vocab(token2idx, vocab_path)

    print("Training complete.")


# ============================================================================
# use()
# ============================================================================

def use(args):
    """Interactive mode: input tokens, see top-10 predicted next tokens."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)

    # Load vocabulary and model.
    vocab_file = '../data/vocab.txt'
    token2idx = load_vocab(vocab_file)
    vocab_size = len(token2idx)
    unk_idx = token2idx['<unk>']
    bos_idx = token2idx['<bos>']

    model, params = load_model('lm_model.pt', token2idx)

    idx2token = {i: t for t, i in token2idx.items()}

    print(f"\nEnter tokens separated by spaces (empty to quit):\n")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            print("Exiting interactive mode.")
            break

        tokens = user_input.lower().split()

        # Build index sequence: <bos> followed by user tokens.
        indices = [bos_idx] + [token2idx.get(t, unk_idx) for t in tokens]
        processed = [idx2token[i] for i in indices]
        print(f"  Processed tokens: {' '.join(processed)}")

        # Run through the Transformer.
        input_tensor = torch.tensor([indices], dtype=torch.long)  # (1, seq_len)

        with torch.no_grad():
            # YOUR CODE HERE: BEGIN
            logits = model(input_tensor)

            # last token prediction
            logits = logits[0, -1]

            probs = torch.softmax(logits, dim=-1)

            # YOUR CODE HERE: END

        # Top 10 predictions.
        top_probs, top_indices = torch.topk(probs, 10)
        print("  Top 10 predictions:")
        for i in range(10):
            tok = idx2token[top_indices[i].item()]
            prob = top_probs[i].item()
            print(f"    {tok:<20s} {prob:.6f}")
        print()


# ============================================================================
# prompt()
# ============================================================================

def generate_next_token(model, indices, greedy, unk_idx):
    """Generate the next token given a sequence of token indices.

    Args:
        model: the TransformerLM model (in eval mode).
        indices: list of integer indices (full sequence so far).
        greedy: if True, pick argmax; if False, sample.
        unk_idx: index of <unk> to suppress.
    Returns:
        The integer index of the generated token.
    """
    input_tensor = torch.tensor([indices], dtype=torch.long)

    with torch.no_grad():
        # YOUR CODE HERE: BEGIN
        logits = model(input_tensor)
        logits = logits[0, -1]

        probs = torch.softmax(logits, dim=-1)

        # avoid <unk>
        probs[unk_idx] = 0
        probs = probs / probs.sum()

        # YOUR CODE HERE: END

    if greedy:
        return torch.argmax(probs).item()
    else:
        return torch.multinomial(probs, num_samples=1).item()


def prompt(args, greedy=True):
    """Autoregressive text generation with the Transformer LM.

    Reads a prompt from the user, generates tokens one at a time.
    Stops at <eos>, sentence-ending punctuation, or after MAX_GEN tokens.
    """
    print("=" * 60)
    print("PROMPT MODE" + (" (greedy)" if greedy else " (sampling)"))
    print("=" * 60)

    STOP_TOKENS = {'.', '!', '?'}
    MAX_GEN = 200

    # Load vocabulary and model.
    vocab_file = '../data/vocab.txt'
    token2idx = load_vocab(vocab_file)
    vocab_size = len(token2idx)
    unk_idx = token2idx['<unk>']
    bos_idx = token2idx['<bos>']
    eos_idx = token2idx['<eos>']

    model, params = load_model('lm_model.pt', token2idx)

    idx2token = {i: t for t, i in token2idx.items()}

    print(f"\nEnter a sequence of tokens to use as prompt (empty to quit):\n")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            print("Exiting prompt mode.")
            break

        tokens = user_input.lower().split()

        # Build index sequence: <bos> followed by user tokens.
        indices = [bos_idx] + [token2idx.get(t, unk_idx) for t in tokens]
        processed = [idx2token[i] for i in indices]
        print(f"  Prompt: {' '.join(processed)}")

        # Autoregressive generation.
        generated = []

        for _ in range(MAX_GEN):
            next_idx = generate_next_token(model, indices, greedy, unk_idx)
            next_token = idx2token[next_idx]

            if next_idx == eos_idx:
                break

            generated.append(next_token)

            if next_token in STOP_TOKENS:
                break

            indices.append(next_idx)

        # Display prompt tokens (without <bos>) and generated tokens.
        prompt_display = ' '.join(processed[1:])  # skip <bos> in display
        print(f"  Generated: {prompt_display} {' '.join(generated)}\n")


# ============================================================================
# evaluate()
# ============================================================================

def evaluate(args):
    """Evaluate the trained model on the test corpus and report perplexity."""
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Load vocabulary and model.
    vocab_file = '../data/vocab.txt'
    token2idx = load_vocab(vocab_file)
    vocab_size = len(token2idx)

    model, params = load_model('lm_model.pt', token2idx)

    # Read test stories and compute loss.
    print(f"Reading test stories: {args.test_corpus}")
    test_stories = read_stories(args.test_corpus, token2idx)
    print(f"  Test stories: {len(test_stories)}")
    total_tokens = sum(len(s) - 1 for s in test_stories)
    print(f"  Test tokens: {total_tokens}")

    # YOUR CODE HERE: BEGIN
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    avg_loss = compute_loss(model, test_stories, criterion)
    perplexity = math.exp(avg_loss)

    # YOUR CODE HERE: END
    
    print(f"\n  Test Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"  Test Perplexity:         {perplexity:.2f}\n")


# ============================================================================
# main()
# ============================================================================

def main():
    """Parse command line arguments and dispatch to train/use/evaluate/prompt."""
    parser = argparse.ArgumentParser(
        description='Transformer Decoder Language Model (TransformerLM)')

    # Mode of operation (mutually exclusive).
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-train', action='store_true', help='Train the model.')
    mode.add_argument('-use', action='store_true', help='Interactive mode.')
    mode.add_argument('-evaluate', action='store_true', help='Evaluate on test corpus.')
    mode.add_argument('-prompt', action='store_true', help='Autoregressive generation.')

    # Transformer architecture hyperparameters.
    parser.add_argument('-num_layers', type=int, default=8,
                        help='Number of Transformer decoder layers (default: 8).')
    parser.add_argument('-d_model', type=int, default=128,
                        help='Model dimensionality (default: 128).')
    parser.add_argument('-num_heads', type=int, default=4,
                        help='Number of attention heads (default: 4).')
    parser.add_argument('-dim_feedforward', type=int, default=512,
                        help='Feed-forward intermediate dimension (default: 512).')
    parser.add_argument('-activation', type=str, default='relu',
                        choices=['relu', 'gelu'],
                        help='FFN activation function (default: relu).')
    parser.add_argument('-dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1).')
    parser.add_argument('-norm_first', action='store_true', default=False,
                        help='Use Pre-LN (LayerNorm before sublayers). Default: Post-LN.')

    # Training hyperparameters.
    parser.add_argument('-E', type=int, default=4,
                        help='Number of training epochs (default: 4).')
    parser.add_argument('-B', type=int, default=10,
                        help='Number of stories per minibatch (default: 10).')
    parser.add_argument('-VSize', type=int, default=10000,
                        help='Vocabulary size (default: 10000).')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001).')

    # File paths.
    parser.add_argument('-train_corpus', type=str,
                        default='../data/tiny/tiny.train.tokens.4M',
                        help='Path to tokenized training corpus.')
    parser.add_argument('-valid_corpus', type=str,
                        default='../data/tiny/tiny.valid.tokens.200K',
                        help='Path to tokenized validation corpus.')
    parser.add_argument('-test_corpus', type=str,
                        default='../data/tiny/tiny.test.tokens.4M',
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
