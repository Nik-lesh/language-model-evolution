import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
from tqdm import tqdm
import pickle

# Add src to path so we can import our models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simple_rnn import SimpleRNN, count_parameters


class CharDataset(Dataset):
    """
    PyTorch Dataset wrapper for our character sequences.
    
    Makes it easy to use with DataLoader for batching.
    """
    
    def __init__(self, encoded_data, seq_length):
        self.data = encoded_data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) // self.seq_length
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        
        return torch.LongTensor(x), torch.LongTensor(y)


def load_dataset(dataset_path='data/dataset.pkl'):
    """Load the preprocessed dataset."""
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(dataset_dict['encoded']):,} characters")
    print(f"✓ Vocabulary size: {dataset_dict['vocab_size']}")
    
    return dataset_dict


def create_dataloaders(dataset_dict, batch_size=64, seq_length=100):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_dict: Dictionary with encoded data and mappings
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence
    
    Returns:
        train_loader, val_loader, dataset_dict
    """
    print(f"\nCreating dataloaders...")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    
    # Split into train/val (90/10)
    split_idx = int(0.9 * len(dataset_dict['encoded']))
    
    train_data = dataset_dict['encoded'][:split_idx]
    val_data = dataset_dict['encoded'][split_idx:]
    
    print(f"  Train size: {len(train_data):,} characters")
    print(f"  Val size: {len(val_data):,} characters")
    
    # Create datasets
    train_dataset = CharDataset(train_data, seq_length)
    val_dataset = CharDataset(val_data, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        Average loss for this epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (x, y) in enumerate(progress_bar):
        # Move to device
        x, y = x.to(device), y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(x)
        
        # Reshape target to match output
        y = y.view(-1)
        
        # Calculate loss
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            output, _ = model(x)
            y = y.view(-1)
            
            loss = criterion(output, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def generate_sample(model, dataset_dict, device, start_text="Money is", length=200):
    """Generate a sample of text."""
    model.eval()
    
    # Encode start text
    chars = [dataset_dict['char_to_idx'][ch] for ch in start_text]
    input_seq = torch.LongTensor(chars).unsqueeze(0).to(device)
    
    generated = start_text
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            
            # Get probabilities for next character
            probs = torch.softmax(output[-1], dim=0)
            
            # Sample next character
            next_idx = torch.multinomial(probs, 1).item()
            next_char = dataset_dict['idx_to_char'][next_idx]
            
            generated += next_char
            input_seq = torch.LongTensor([[next_idx]]).to(device)
    
    return generated


def train(model_name='simple_rnn', 
          epochs=50, 
          batch_size=64, 
          seq_length=100,
          learning_rate=0.002,
          embedding_dim=128,
          hidden_dim=256,
          num_layers=2):
    """
    Main training function.
    
    Args:
        model_name: Name of model (for saving)
        epochs: Number of training epochs
        batch_size: Batch size
        seq_length: Sequence length
        learning_rate: Learning rate for optimizer
        embedding_dim: Size of embeddings
        hidden_dim: Size of hidden state
        num_layers: Number of RNN layers
    """
    
    print("=" * 60)
    print("TRAINING SIMPLE RNN")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    dataset_dict = load_dataset('data/dataset.pkl')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_dict, 
        batch_size=batch_size, 
        seq_length=seq_length
    )
    
    # Create model
    print(f"\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    
    model = SimpleRNN(
        vocab_size=dataset_dict['vocab_size'],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
    )
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print(f"\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60 + "\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print stats
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time: {time.time() - start_time:.2f}s")
        
        # Generate sample every 5 epochs
        if epoch % 5 == 0:
            print("\n" + "-" * 60)
            print("Sample generation:")
            print("-" * 60)
            sample = generate_sample(model, dataset_dict, device)
            print(sample)
            print("-" * 60 + "\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': dataset_dict['vocab_size'],
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers
            }
            torch.save(checkpoint, f'checkpoints/{model_name}_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final model
    torch.save(checkpoint, f'checkpoints/{model_name}_final.pth')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(f'checkpoints/{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: checkpoints/{model_name}_best.pth")


if __name__ == "__main__":
    # Train with default parameters
    train(
        model_name='simple_rnn',
        epochs=50,
        batch_size=64,
        seq_length=100,
        learning_rate=0.002,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )