import torch
import pickle
import sys
from pathlib import Path
from typing import Optional
import logging
from huggingface_hub import hf_hub_download
import os

# Fix path to find src/models/
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent  # backend/
project_root = backend_dir.parent  # language-model-evolution/
sys.path.insert(0, str(project_root))

# Now this will work
from src.models.transformer import TransformerModel
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class ModelService:
    """
    Singleton service for model inference.
    
    Loads models once on startup, reuses for all requests.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
        
        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = True
        
        logger.info(f"Model service initialized on device: {self.device}")
    
    def load(self):
        """Load model and dataset."""
        
        logger.info("Loading from HuggingFace Hub...")
    
        repo_id = "Nikilesh9/financial-language-model"
        
        try:
            logger.info("Downloading dataset...")
            dataset_path = hf_hub_download(
                repo_id=repo_id,
                filename="mega_word_dataset.pkl",
                cache_dir="./hf_cache"
            )
        
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
        
            logger.info(f"✅ Dataset loaded (vocab: {self.dataset['vocab_size']:,})")
        
        # Download model
            logger.info("Downloading model...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="transformer_1gb_balanced_best.pth",
                cache_dir="./hf_cache"
            )
            checkpoint = torch.load(model_path, map_location=self.device)
        
            self.model = TransformerModel(
                vocab_size=self.dataset['vocab_size'],
                d_model=512,
                nhead=8,
                num_layers=6,
                dim_feedforward=2048,
                dropout=0.2
                )
        
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
        
            logger.info(f"✅ Model loaded on {self.device}")
        
        # Warm up
            self.generate("test", max_length=10)
            logger.info("✅ Model ready!")
        
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 100, 
                 temperature: float = 0.8) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Starting text
            max_length: Number of words to generate
            temperature: Sampling temperature (0.1-2.0)
        
        Returns:
            Generated text
        """
        
        if self.model is None or self.dataset is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Tokenize prompt
        words = prompt.lower().split()
        unk_idx = self.dataset['word_to_idx'].get('<UNK>', 1)
        
        input_seq = [self.dataset['word_to_idx'].get(w, unk_idx) for w in words]
        input_seq = torch.LongTensor([input_seq]).to(self.device)
        
        generated = words.copy()
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(input_seq)
                
                # Apply temperature
                logits = output[-1] / temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sample next word
                next_idx = torch.multinomial(probs, 1).item()
                next_word = self.dataset['idx_to_word'][next_idx]
                if next_word in ['<UNK>', '<PAD>', '<EOS>']:
                    # Fallback: pick most likely valid word
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    for idx in sorted_indices[:10]:
                        word = self.dataset['idx_to_word'][idx.item()]
                        if word not in ['<UNK>', '<PAD>', '<EOS>', ',', '.']:
                            next_word = word
                            next_idx = idx.item()
                            break
                
                generated.append(next_word)
                
                # Update input
                next_tensor = torch.LongTensor([[next_idx]]).to(self.device)
                input_seq = torch.cat([input_seq, next_tensor], dim=1)
                
                # Keep context window (last 50 words)
                if input_seq.size(1) > 50:
                    input_seq = input_seq[:, -50:]
        
        return ' '.join(generated)
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        
        return {
            "name": "Transformer (1GB Balanced Corpus)",
            "architecture": "6-layer Transformer",
            "parameters": "~12M",
            "vocab_size": self.dataset['vocab_size'] if self.dataset else None,
            "val_loss": 4.01,
            "training_data": "1 GB balanced financial corpus",
            "modern_content_pct": 87,
            "device": str(self.device)
        }

# Global instance
model_service = ModelService()