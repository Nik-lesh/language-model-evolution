import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
from tqdm import tqdm
import pickle
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm import LSTMModel
from models.transformer import TransformerModel

class WordDataset(Dataset):
    """PyTorch Dataset for word-level sequences."""
    
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


def load_word_dataset(dataset_path='data/word_dataset.pkl'):
    """Load word-level dataset."""
    print(f"Loading word-level dataset from {dataset_path}...")
    
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(dataset_dict['encoded']):,} words")
    print(f"✓ Vocabulary size: {dataset_dict['vocab_size']:,}")
    
    return dataset_dict


def create_dataloaders(dataset_dict, batch_size=32, seq_length=50):
    """Create train and validation dataloaders."""
    print(f"\nCreating dataloaders...")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length} words")
    
    # Split into train/val (90/10)
    split_idx = int(0.9 * len(dataset_dict['encoded']))
    
    train_data = dataset_dict['encoded'][:split_idx]
    val_data = dataset_dict['encoded'][split_idx:]
    
    print(f"  Train size: {len(train_data):,} words")
    print(f"  Val size: {len(val_data):,} words")
    
    # Create datasets
    train_dataset = WordDataset(train_data, seq_length)
    val_dataset = WordDataset(val_data, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'transformer':
            output = model(x)
        else:
            output, _ = model(x)
        
        # Calculate loss
        y = y.view(-1)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, model_type):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            if model_type == 'transformer':
                output = model(x)
            else:
                output, _ = model(x)
            
            y = y.view(-1)
            loss = criterion(output, y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def generate_sample(model, dataset_dict, device, start_text="Money is", length=50, model_type='lstm'):
    """Generate sample text."""
    model.eval()
    
    # Tokenize start text
    words = start_text.lower().split()
    unk_idx = dataset_dict['word_to_idx'].get('<UNK>', 1)
    
    input_seq = [dataset_dict['word_to_idx'].get(w, unk_idx) for w in words]
    input_seq = torch.LongTensor([input_seq]).to(device)
    
    generated_words = words.copy()
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            if model_type == 'transformer':
                output = model(input_seq)
            else:
                output, hidden = model(input_seq, hidden)
            
            # Get last prediction
            probs = torch.softmax(output[-1], dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            
            next_word = dataset_dict['idx_to_word'][next_idx]
            generated_words.append(next_word)
            
            # Update input
            input_seq = torch.LongTensor([[next_idx]]).to(device)
    
    return ' '.join(generated_words)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model_type='lstm', epochs=50, batch_size=32, seq_length=50, learning_rate=0.001):
    """Main training function for word-level models."""
    
    print("=" * 60)
    print(f"TRAINING {model_type.upper()} - WORD LEVEL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    dataset_dict = load_word_dataset('data/word_dataset.pkl')
    
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
    
    vocab_size = dataset_dict['vocab_size']
    
    if model_type == 'lstm':
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2,
            dropout=0.3
        ).to(device)
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.2
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print(f"\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length} words")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60 + "\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_type)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, model_type)
        
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
        
        # Generate sample every 10 epochs
        if epoch % 10 == 0:
            print("\n" + "-" * 60)
            print("Sample generation:")
            print("-" * 60)
            sample = generate_sample(model, dataset_dict, device, model_type=model_type)
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
                'vocab_size': vocab_size,
                'model_type': model_type
            }
            torch.save(checkpoint, f'checkpoints/{model_type}_word_level_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(f'checkpoints/{model_type}_word_level_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'lstm'
    
    if model_type == 'lstm':
        train(
            model_type='lstm',
            epochs=50,
            batch_size=32,
            seq_length=50,
            learning_rate=0.001
        )
    elif model_type == 'transformer':
        train(
            model_type='transformer',
            epochs=50,
            batch_size=32,
            seq_length=50,
            learning_rate=0.0005
        )
    else:
        print(f"Unknown model: {model_type}")
        print("Usage: python src/train_word_level.py [lstm|transformer]")