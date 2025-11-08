import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    Simple Recurrent Neural Network for character-level language modeling.
    
    Architecture:
    1. Embedding layer: Converts character indices to dense vectors
    2. RNN layers: Process sequences and maintain hidden state
    3. Fully connected layer: Predicts next character
    
    This is the SIMPLEST model - we'll compare it to LSTM and Transformer.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size: Number of unique characters (113 in our case)
            embedding_dim: Size of character embeddings (how we represent each char)
            hidden_dim: Size of RNN hidden state (the "memory")
            num_layers: Number of stacked RNN layers
            dropout: Dropout probability (prevents overfitting)
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer: character index -> dense vector
        # Example: character 'a' (index 0) -> [0.23, -0.45, 0.12, ...]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        # Takes sequence of embeddings and produces sequence of hidden states
        self.rnn = nn.RNN(
            input_size=embedding_dim,    # Size of input vectors (embeddings)
            hidden_size=hidden_dim,       # Size of hidden state
            num_layers=num_layers,        # Stack multiple RNN layers
            batch_first=True,             # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers
        )
        
        # Fully connected layer: hidden state -> character probabilities
        # Output: probability distribution over all possible next characters
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input sequences (batch_size, seq_length)
               Contains character indices
            hidden: Previous hidden state (optional)
                   If None, initializes to zeros
        
        Returns:
            output: Predicted logits for next character (batch_size * seq_length, vocab_size)
            hidden: Updated hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 1. Embed the input characters
        # x: (batch_size, seq_length) -> embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # 2. Pass through RNN
        # output: (batch_size, seq_length, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        output, hidden = self.rnn(embedded, hidden)
        
        # 3. Apply dropout
        output = self.dropout(output)
        
        # 4. Reshape for fully connected layer
        # Flatten: (batch_size, seq_length, hidden_dim) -> (batch_size * seq_length, hidden_dim)
        output = output.contiguous().view(-1, self.hidden_dim)
        
        # 5. Predict next character
        # (batch_size * seq_length, hidden_dim) -> (batch_size * seq_length, vocab_size)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state to zeros.
        
        Args:
            batch_size: Size of the batch
            device: CPU or CUDA device
        
        Returns:
            Hidden state tensor: (num_layers, batch_size, hidden_dim)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    
    def generate(self, dataset, start_text="Money is", length=200, temperature=0.8):
        """
        Generate text using the trained model.
        
        Args:
            dataset: TextDataset object (for encoding/decoding)
            start_text: Text to start generation with
            length: How many characters to generate
            temperature: Controls randomness (higher = more random)
                        0.5 = conservative, 1.0 = balanced, 1.5 = creative
        
        Returns:
            Generated text string
        """
        self.eval()  # Set to evaluation mode
        if isinstance(dataset, dict):
            char_to_idx = dataset['char_to_idx']
            idx_to_char = dataset['idx_to_char']
        else:
            char_to_idx = dataset.char_to_idx
            idx_to_char = dataset.idx_to_char
        
        # Encode the starting text
        chars = [dataset.char_to_idx[ch] for ch in start_text]
        input_seq = torch.LongTensor(chars).unsqueeze(0)  # Add batch dimension
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_seq = input_seq.to(device)
        
        # Initialize hidden state
        hidden = None
        
        # Generated text starts with the input
        generated = start_text
        
        with torch.no_grad():  # No gradients needed for generation
            for _ in range(length):
                # Forward pass
                output, hidden = self.forward(input_seq, hidden)
                
                # Get the last output (prediction for next character)
                output = output[-1] / temperature  # Apply temperature
                
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
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMPLE RNN MODEL")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 113  # From our dataset
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    
    # Create model
    model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    seq_length = 100
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    output, hidden = model(x)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Hidden shape: {hidden.shape}")
    
    # Verify output shape
    expected_output_shape = (batch_size * seq_length, vocab_size)
    assert output.shape == expected_output_shape, f"Wrong output shape! Expected {expected_output_shape}"
    
    print(f"\n✓ Model test passed!")
    print("=" * 60)