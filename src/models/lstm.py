import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) for character-level language modeling.
    
    Key Improvements over Simple RNN:
    1. Memory Cell: Can store information for long periods
    2. Gates: Control what to remember, forget, and output
       - Forget Gate: Decides what to throw away from cell state
       - Input Gate: Decides what new information to store
       - Output Gate: Decides what to output based on cell state
    3. Better gradient flow: Solves vanishing gradient problem
    
    This should generate MORE COHERENT text than Simple RNN!
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size: Number of unique characters (113 in our case)
            embedding_dim: Size of character embeddings (larger than RNN!)
            hidden_dim: Size of LSTM hidden state (larger than RNN!)
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (prevents overfitting)
        
        Note: We use larger hidden_dim than RNN because LSTM can handle it better!
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer: character index -> dense vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer (the magic happens here!)
        # LSTM has 4x more parameters than RNN due to gates
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer: hidden state -> character probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input sequences (batch_size, seq_length)
            hidden: Tuple of (h_0, c_0) - hidden state and cell state
                   If None, initializes to zeros
        
        Returns:
            output: Predicted logits (batch_size * seq_length, vocab_size)
            hidden: Updated (hidden_state, cell_state) tuple
        
        Key difference from RNN:
        - LSTM returns TWO states: hidden state AND cell state
        - Cell state is the "long-term memory"
        - Hidden state is the "short-term memory"
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 1. Embed the input characters
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # 2. Pass through LSTM
        # LSTM internally computes:
        #   - Forget gate: what to remove from cell state
        #   - Input gate: what new info to add to cell state
        #   - Output gate: what to output from cell state
        output, hidden = self.lstm(embedded, hidden)
        
        # 3. Apply dropout
        output = self.dropout(output)
        
        # 4. Reshape for fully connected layer
        output = output.contiguous().view(-1, self.hidden_dim)
        
        # 5. Predict next character
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and cell state to zeros.
        
        Returns:
            Tuple of (hidden_state, cell_state)
            Both are: (num_layers, batch_size, hidden_dim)
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(self, dataset, start_text="Money is", length=200, temperature=0.8):
        """
        Generate text using the trained LSTM.
        
        Args:
            dataset: TextDataset dict (for encoding/decoding)
            start_text: Text to start generation with
            length: How many characters to generate
            temperature: Controls randomness
                        Lower (0.5) = more conservative, repetitive
                        Higher (1.5) = more creative, random
        
        Returns:
            Generated text string
        """
        self.eval()

        if isinstance(dataset, dict):
            char_to_idx = dataset['char_to_idx']
            idx_to_char = dataset['idx_to_char']
        else:
            char_to_idx = dataset.char_to_idx
            idx_to_char = dataset.idx_to_char
        
        # Encode the starting text
        chars = [dataset['char_to_idx'][ch] for ch in start_text]
        input_seq = torch.LongTensor(chars).unsqueeze(0)
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_seq = input_seq.to(device)
        
        # Initialize hidden state
        hidden = None
        
        # Generated text starts with the input
        generated = start_text
        
        with torch.no_grad():
            for _ in range(length):
                # Forward pass
                output, hidden = self.forward(input_seq, hidden)
                
                # Get the last output (prediction for next character)
                output = output[-1] / temperature
                
                # Convert to probabilities
                probs = torch.softmax(output, dim=0)
                
                # Sample from the probability distribution
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # Decode and add to generated text
                next_char = idx_to_char[next_char_idx]
                generated += next_char
                
                # Use this character as input for next iteration
                input_seq = torch.LongTensor([[next_char_idx]]).to(device)
        
        return generated


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LSTM MODEL")
    print("=" * 60)
    
    # Model parameters (larger than RNN!)
    vocab_size = 113
    embedding_dim = 256  # Doubled from RNN
    hidden_dim = 512     # Doubled from RNN
    num_layers = 2
    
    # Create model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nModel Parameters:")
    rnn_params = 273_905  # From our Simple RNN
    lstm_params = count_parameters(model)
    print(f"  Total parameters: {lstm_params:,}")
    print(f"  vs Simple RNN: {rnn_params:,}")
    print(f"  Difference: {lstm_params - rnn_params:,} ({(lstm_params/rnn_params - 1)*100:.1f}% more)")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    seq_length = 100
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    output, (hidden, cell) = model(x)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Cell shape:   {cell.shape}")
    
    # Verify output shape
    expected_output_shape = (batch_size * seq_length, vocab_size)
    assert output.shape == expected_output_shape, f"Wrong output shape!"
    
    print(f"\n✓ Model test passed!")
    print("\nKey Differences from RNN:")
    print("  ✓ Has cell state (long-term memory)")
    print("  ✓ Uses gates for controlled information flow")
    print("  ✓ Better at learning long-range dependencies")
    print("  ✓ More parameters but trains almost as fast")
    print("=" * 60)