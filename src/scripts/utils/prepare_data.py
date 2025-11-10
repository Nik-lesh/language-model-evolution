import numpy as np
import pickle
import os
from collections import Counter

class TextDataset:
    """
    Prepare text data for character-level language modeling.
    
    What this class does:
    1. Reads text file
    2. Creates character-to-index mappings (vocabulary)
    3. Encodes all text as numbers
    4. Creates training sequences
    5. Generates batches for training
    """
    
    def __init__(self, filepath, seq_length=100):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path to text file
            seq_length: How many characters in each training sequence
                       (like: "money is important" = 18 characters)
        """
        print("=" * 60)
        print("PREPARING DATASET")
        print("=" * 60)
        
        self.seq_length = seq_length
        
        # Read the text
        print(f"\nReading: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print(f"✓ Loaded {len(self.text):,} characters")
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Encode the text
        self._encode_text()
        
        # Calculate dataset statistics
        self._print_stats()
    
    def _build_vocabulary(self):
        """
        Create mappings between characters and numbers.
        
        Example:
        'a' -> 0
        'b' -> 1
        'c' -> 2
        ...
        
        This is our "vocabulary" - all unique characters in the text.
        """
        print("\nBuilding vocabulary...")
        
        # Get all unique characters and sort them
        chars = sorted(list(set(self.text)))
        
        self.vocab_size = len(chars)
        self.chars = chars
        
        # Create character -> index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        
        # Create index -> character mapping (for decoding later)
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"✓ Vocabulary size: {self.vocab_size} unique characters")
        print(f"  Characters: {''.join(chars[:50])}...")
    
    def _encode_text(self):
        """
        Convert text to numbers using our vocabulary.
        
        Example:
        "cat" -> [2, 0, 19] (using char_to_idx mappings)
        """
        print("\nEncoding text...")
        
        # Convert each character to its index
        self.encoded = np.array([self.char_to_idx[ch] for ch in self.text])
        
        print(f"✓ Encoded {len(self.encoded):,} characters")
        
        # Show example
        sample_text = self.text[:50]
        sample_encoded = self.encoded[:50]
        print(f"\nExample encoding:")
        print(f"  Text: {sample_text}")
        print(f"  Encoded: {sample_encoded.tolist()[:20]}...")
    
    def _print_stats(self):
        """Print useful statistics about the dataset."""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # Calculate how many sequences we can make
        num_sequences = len(self.encoded) // self.seq_length
        
        print(f"\nSequence length: {self.seq_length} characters")
        print(f"Total sequences: {num_sequences:,}")
        print(f"Total characters: {len(self.encoded):,}")
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Character frequency
        char_counts = Counter(self.text)
        print(f"\nMost common characters:")
        for char, count in char_counts.most_common(10):
            # Show readable character name
            if char == ' ':
                char_display = '<space>'
            elif char == '\n':
                char_display = '<newline>'
            else:
                char_display = char
            print(f"  '{char_display}': {count:,} ({100 * count / len(self.text):.2f}%)")
    
    def get_batches(self, batch_size, split='train'):
        """
        Generate batches of training sequences.
        
        Args:
            batch_size: How many sequences per batch
            split: 'train' or 'val' (validation)
        
        Yields:
            x: Input sequences (batch_size, seq_length)
            y: Target sequences (batch_size, seq_length)
               Target is input shifted by 1 (predict next character)
        
        How it works:
        Input:  "hello worl"
        Target: "ello world"
        
        Model learns: given "hello worl", predict "d"
        """
        
        # Split into train/val (90/10 split)
        split_idx = int(0.9 * len(self.encoded))
        
        if split == 'train':
            data = self.encoded[:split_idx]
        else:
            data = self.encoded[split_idx:]
        
        # Calculate how many batches we can make
        num_batches = len(data) // (batch_size * self.seq_length)
        
        # Trim data to fit batches evenly
        data = data[:num_batches * batch_size * self.seq_length]
        
        # Reshape into (batch_size, num_sequences * seq_length)
        data = data.reshape(batch_size, -1)
        
        # Generate batches
        for i in range(0, data.shape[1] - self.seq_length, self.seq_length):
            # Input sequences
            x = data[:, i:i + self.seq_length]
            
            # Target sequences (shifted by 1)
            y = data[:, i + 1:i + self.seq_length + 1]
            
            yield x, y
    
    def decode(self, indices):
        """
        Convert numbers back to text.
        
        Args:
            indices: Array of character indices
        
        Returns:
            String of decoded text
        """
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def save(self, filename='data/dataset.pkl'):
        """Save the dataset for later use."""
        print(f"\nSaving dataset to {filename}...")
        
        # Save everything we need
        dataset_dict = {
            'text': self.text,
            'encoded': self.encoded,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'chars': self.chars,
            'seq_length': self.seq_length
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"✓ Saved! ({os.path.getsize(filename) / 1024:.2f} KB)")
    
    @classmethod
    def load(cls, filename='data/dataset.pkl'):
        """Load a previously saved dataset."""
        print(f"Loading dataset from {filename}...")
        
        with open(filename, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        # Create new instance
        dataset = cls.__new__(cls)
        
        # Restore all attributes
        dataset.text = dataset_dict['text']
        dataset.encoded = dataset_dict['encoded']
        dataset.char_to_idx = dataset_dict['char_to_idx']
        dataset.idx_to_char = dataset_dict['idx_to_char']
        dataset.vocab_size = dataset_dict['vocab_size']
        dataset.chars = dataset_dict['chars']
        dataset.seq_length = dataset_dict['seq_length']
        
        print(f"✓ Loaded dataset with {len(dataset.encoded):,} characters")
        
        return dataset


def test_dataset():
    """Test the dataset to make sure everything works."""
    print("\n" + "=" * 60)
    print("TESTING DATASET")
    print("=" * 60)
    
    # Create dataset
    dataset = TextDataset('../../data/training_corpus_clean.txt', seq_length=100)
    
    # Test batches
    print("\n" + "=" * 60)
    print("TESTING BATCH GENERATION")
    print("=" * 60)
    
    batch_size = 4
    print(f"\nGenerating a test batch (batch_size={batch_size})...")
    
    # Get one batch
    batches = dataset.get_batches(batch_size, split='train')
    x, y = next(batches)
    
    print(f"\nBatch shapes:")
    print(f"  Input (x):  {x.shape}")
    print(f"  Target (y): {y.shape}")
    
    # Show first sequence
    print(f"\nFirst sequence in batch:")
    print(f"  Input text:  {dataset.decode(x[0][:50])}...")
    print(f"  Target text: {dataset.decode(y[0][:50])}...")
    
    # Verify input and target are shifted correctly
    print(f"\nVerifying shift:")
    print(f"  Input char 0:  '{dataset.idx_to_char[x[0][0]]}'")
    print(f"  Target char 0: '{dataset.idx_to_char[y[0][0]]}' (should be char 1 from input)")
    
    # Test save/load
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD")
    print("=" * 60)
    
    dataset.save('../../data/dataset.pkl')
    
    loaded_dataset = TextDataset.load('../../data/dataset.pkl')
    
    print(f"\nVerifying loaded dataset:")
    print(f"  Original vocab size: {dataset.vocab_size}")
    print(f"  Loaded vocab size:   {loaded_dataset.vocab_size}")
    print(f"  Match: {'✓' if dataset.vocab_size == loaded_dataset.vocab_size else '✗'}")


if __name__ == "__main__":
    # Run the test
    test_dataset()
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    
    # Save the dataset
    print("The processed data is saved in 'data/dataset.pkl'")