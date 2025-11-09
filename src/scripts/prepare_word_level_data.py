import numpy as np
import pickle
import os
import re
from collections import Counter

class WordLevelDataset:
    """
    Prepare text data for WORD-level language modeling.
    
    Key differences from character-level:
    - Tokens are words instead of characters
    - Smaller vocabulary (10K words vs 113 chars)
    - Better for Transformers (designed for word-level)
    - More semantic meaning per token
    """
    
    def __init__(self, filepath, seq_length=50, max_vocab=10000):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path to text file
            seq_length: How many WORDS in each sequence (not chars!)
            max_vocab: Maximum vocabulary size (most common words)
        """
        print("=" * 60)
        print("PREPARING WORD-LEVEL DATASET")
        print("=" * 60)
        
        self.seq_length = seq_length
        self.max_vocab = max_vocab
        
        # Read the text
        print(f"\nReading: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print(f"✓ Loaded {len(self.text):,} characters")
        
        # Tokenize into words
        self._tokenize_words()
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Encode the text
        self._encode_text()
        
        # Calculate dataset statistics
        self._print_stats()
    
    def _tokenize_words(self):
        """
        Convert text to list of words.
        
        Tokenization strategy:
        - Lowercase everything
        - Split on whitespace
        - Keep punctuation as separate tokens
        - Handle contractions (don't → do n't)
        """
        print("\nTokenizing into words...")
        
        # Lowercase
        text = self.text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?,;:\(\)\[\]{}"\'])', r' \1 ', text)
        
        # Split on whitespace
        words = text.split()
        
        # Remove empty strings
        self.words = [w for w in words if w.strip()]
        
        print(f"✓ Tokenized into {len(self.words):,} words")
    
    def _build_vocabulary(self):
        """
        Create vocabulary of most common words.
        
        Strategy:
        - Keep top max_vocab most common words
        - Add special tokens: <PAD>, <UNK>, <EOS>
        - Map rare words to <UNK>
        """
        print("\nBuilding vocabulary...")
        
        # Count word frequencies
        word_counts = Counter(self.words)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_vocab - 3)
        
        # Build vocabulary with special tokens
        vocab_words = ['<PAD>', '<UNK>', '<EOS>']  # Special tokens first
        vocab_words += [word for word, _ in most_common]
        
        self.vocab_size = len(vocab_words)
        self.vocab_words = vocab_words
        
        # Create mappings
        self.word_to_idx = {word: i for i, word in enumerate(vocab_words)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab_words)}
        
        print(f"✓ Vocabulary size: {self.vocab_size:,} words")
        print(f"  Special tokens: <PAD>=0, <UNK>=1, <EOS>=2")
        print(f"  Top 10 words: {vocab_words[3:13]}")
        
        # Calculate coverage
        known_words = sum(1 for w in self.words if w in self.word_to_idx)
        coverage = 100 * known_words / len(self.words)
        print(f"  Vocabulary coverage: {coverage:.2f}%")
    
    def _encode_text(self):
        """
        Convert words to indices using vocabulary.
        
        Maps each word to its index.
        Unknown words → <UNK> token (index 1)
        """
        print("\nEncoding text...")
        
        unk_idx = self.word_to_idx['<UNK>']
        
        # Convert each word to its index
        self.encoded = np.array([
            self.word_to_idx.get(word, unk_idx) 
            for word in self.words
        ])
        
        print(f"✓ Encoded {len(self.encoded):,} words")
        
        # Show example
        sample_words = self.words[:20]
        sample_encoded = self.encoded[:20]
        print(f"\nExample encoding:")
        print(f"  Words: {' '.join(sample_words[:10])}...")
        print(f"  Indices: {sample_encoded[:10].tolist()}...")
    
    def _print_stats(self):
        """Print useful statistics about the dataset."""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # Calculate how many sequences we can make
        num_sequences = len(self.encoded) // self.seq_length
        
        print(f"\nSequence length: {self.seq_length} words")
        print(f"Total sequences: {num_sequences:,}")
        print(f"Total words: {len(self.encoded):,}")
        print(f"Total characters: {len(self.text):,}")
        print(f"Vocabulary size: {self.vocab_size:,}")
        
        # Average word length
        avg_word_len = len(self.text) / len(self.words)
        print(f"Average word length: {avg_word_len:.2f} characters")
        
        # Most common words
        word_counts = Counter(self.words)
        print(f"\nMost common words:")
        for word, count in word_counts.most_common(15):
            print(f"  '{word}': {count:,} ({100*count/len(self.words):.2f}%)")
    
    def get_batches(self, batch_size, split='train'):
        """
        Generate batches of training sequences.
        
        Args:
            batch_size: How many sequences per batch
            split: 'train' or 'val' (validation)
        
        Yields:
            x: Input sequences (batch_size, seq_length)
            y: Target sequences (batch_size, seq_length)
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
        Convert indices back to text.
        
        Args:
            indices: Array of word indices
        
        Returns:
            String of decoded text
        """
        words = [self.idx_to_word[idx] for idx in indices]
        return ' '.join(words)
    
    def save(self, filename='data/word_dataset.pkl'):
        """Save the dataset for later use."""
        print(f"\nSaving dataset to {filename}...")
        
        dataset_dict = {
            'text': self.text,
            'words': self.words,
            'encoded': self.encoded,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'vocab_words': self.vocab_words,
            'seq_length': self.seq_length,
            'max_vocab': self.max_vocab
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"✓ Saved! ({os.path.getsize(filename) / 1024:.2f} KB)")
    
    @classmethod
    def load(cls, filename='data/word_dataset.pkl'):
        """Load a previously saved dataset."""
        print(f"Loading dataset from {filename}...")
        
        with open(filename, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        # Create new instance
        dataset = cls.__new__(cls)
        
        # Restore all attributes
        for key, value in dataset_dict.items():
            setattr(dataset, key, value)
        
        print(f"✓ Loaded dataset with {len(dataset.encoded):,} words")
        
        return dataset


def test_word_dataset():
    """Test the word-level dataset."""
    print("\n" + "=" * 60)
    print("TESTING WORD-LEVEL DATASET")
    print("=" * 60)
    
    # Create dataset
    dataset = WordLevelDataset(
        'data/training_corpus_clean.txt', 
        seq_length=50,  # 50 words per sequence
        max_vocab=10000
    )
    
    # Test batches
    print("\n" + "=" * 60)
    print("TESTING BATCH GENERATION")
    print("=" * 60)
    
    batch_size = 32
    print(f"\nGenerating a test batch (batch_size={batch_size})...")
    
    # Get one batch
    batches = dataset.get_batches(batch_size, split='train')
    x, y = next(batches)
    
    print(f"\nBatch shapes:")
    print(f"  Input (x):  {x.shape}")
    print(f"  Target (y): {y.shape}")
    
    # Show first sequence
    print(f"\nFirst sequence in batch:")
    print(f"  Input text:  {dataset.decode(x[0][:20])}...")
    print(f"  Target text: {dataset.decode(y[0][:20])}...")
    
    # Test save/load
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD")
    print("=" * 60)
    
    dataset.save('data/word_dataset.pkl')
    
    loaded_dataset = WordLevelDataset.load('data/word_dataset.pkl')
    
    print(f"\nVerifying loaded dataset:")
    print(f"  Original vocab size: {dataset.vocab_size}")
    print(f"  Loaded vocab size:   {loaded_dataset.vocab_size}")
    print(f"  Match: {'✓' if dataset.vocab_size == loaded_dataset.vocab_size else '✗'}")
    
    return dataset


if __name__ == "__main__":
    # Run the test
    dataset = test_word_dataset()
    
    print("\n" + "=" * 60)
    print("WORD-LEVEL DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nKey differences from character-level:")
    print("  ✓ Vocabulary: ~10,000 words vs 113 characters")
    print("  ✓ Sequence length: 50 words vs 100 characters")
    print("  ✓ More semantic meaning per token")
    print("  ✓ Better for Transformer architecture")
    print("\nYou can now train models on word-level data!")