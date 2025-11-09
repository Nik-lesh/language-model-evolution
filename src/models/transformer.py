import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings.
    
    Why we need this:
    - Transformers process all positions in parallel
    - Without this, "Money is good" = "good is Money" (order is lost!)
    - Positional encoding tells the model WHERE each character is
    
    Uses sinusoidal functions (sine and cosine waves) to encode position.
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term for sine/cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with positional information added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer for character-level language modeling.
    
    Architecture:
    1. Embedding: Convert character indices to vectors
    2. Positional Encoding: Add position information
    3. Transformer Encoder: Multi-head self-attention + feed-forward
    4. Output: Predict next character
    
    Key Innovation: ATTENTION
    - Each position attends to ALL other positions
    - Learns which characters are relevant to predict next character
    - Parallel processing (much faster than RNN/LSTM)
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.2):
        """
        Initialize Transformer model.
        
        Args:
            vocab_size: Number of unique characters (113)
            d_model: Dimension of embeddings (must be divisible by nhead!)
            nhead: Number of attention heads (parallel attention mechanisms)
            num_layers: Number of transformer encoder layers
            dim_feedforward: Hidden dimension in feed-forward network
            dropout: Dropout probability
        
        Key constraint: d_model must be divisible by nhead!
        - 8 heads × 32 dimensions = 256 d_model ✓
        - This allows each head to look at different aspects
        """
        super(TransformerModel, self).__init__()
        
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) format
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask to prevent looking at future tokens.
        
        Why we need this:
        When predicting character at position i, we can only look at
        positions 0 to i-1 (not future positions!)
        
        Creates upper triangular matrix of -inf (blocked positions):
        [[0, -inf, -inf],
         [0,    0, -inf],
         [0,    0,    0]]
        
        Position 0 sees only itself
        Position 1 sees positions 0-1
        Position 2 sees positions 0-2
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, x):
        """
        Forward pass through transformer.
        
        Args:
            x: Input sequences (batch_size, seq_length)
               Contains character indices
        
        Returns:
            output: Predicted logits (batch_size * seq_length, vocab_size)
        
        Steps:
        1. Embed characters → vectors
        2. Add positional encoding
        3. Apply self-attention (THE MAGIC!)
        4. Feed through output layer
        """
        # Get batch size and sequence length
        batch_size, seq_len = x.size()
        
        # 1. Embed the input
        # (batch, seq) → (batch, seq, d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Create causal mask (prevent looking at future)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. Pass through transformer encoder
        # Self-attention happens here!
        # Each position attends to all previous positions
        x = self.transformer_encoder(x, mask=mask)
        
        # 5. Reshape for output layer
        x = x.contiguous().view(-1, self.d_model)
        
        # 6. Predict next character
        output = self.fc_out(x)
        
        return output
    
    def generate(self, dataset, start_text="Money is", length=200, temperature=0.8):
        """
        Generate text using the trained transformer.
        
        Args:
            dataset: Dataset dict (for encoding/decoding)
            start_text: Text to start generation with
            length: How many characters to generate
            temperature: Controls randomness
        
        Returns:
            Generated text string
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Handle both dict and object format
        if isinstance(dataset, dict):
            char_to_idx = dataset['char_to_idx']
            idx_to_char = dataset['idx_to_char']
        else:
            char_to_idx = dataset.char_to_idx
            idx_to_char = dataset.idx_to_char
        
        # Encode starting text
        chars = [char_to_idx[ch] for ch in start_text]
        input_seq = torch.LongTensor([chars]).to(device)
        
        generated = start_text
        
        with torch.no_grad():
            for _ in range(length):
                # Forward pass
                output = self.forward(input_seq)
                
                # Get prediction for last position
                output = output[-1] / temperature
                probs = torch.softmax(output, dim=0)
                
                # Sample next character
                next_idx = torch.multinomial(probs, 1).item()
                next_char = idx_to_char[next_idx]
                
                generated += next_char
                
                # Append to sequence
                next_tensor = torch.LongTensor([[next_idx]]).to(device)
                input_seq = torch.cat([input_seq, next_tensor], dim=1)
                
                # Keep sequence length manageable (last 100 chars)
                if input_seq.size(1) > 100:
                    input_seq = input_seq[:, -100:]
        
        return generated


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRANSFORMER MODEL")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 113
    d_model = 256       # Embedding dimension
    nhead = 8           # Number of attention heads
    num_layers = 4      # Number of transformer layers
    dim_feedforward = 1024  # Feed-forward hidden dimension
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nModel Parameters:")
    params = count_parameters(model)
    print(f"  Total parameters: {params:,}")
    print(f"  vs Simple RNN: 273,905")
    print(f"  vs LSTM: 3,765,105")
    print(f"  Transformer: {params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    seq_length = 100
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"  Output shape: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size * seq_length, vocab_size)
    assert output.shape == expected_shape, f"Wrong shape! Expected {expected_shape}"
    
    print(f"\n✓ Model test passed!")
    
    print("\nKey Features:")
    print("  ✓ Self-attention mechanism")
    print("  ✓ Positional encoding")
    print("  ✓ Parallel processing")
    print("  ✓ Multi-head attention")
    print("  ✓ Causal masking (no future peeking)")
    
    print(f"\nAttention Heads: {nhead}")
    print(f"  Each head attends to different patterns")
    print(f"  Head dimension: {d_model // nhead}")
    
    print("=" * 60)