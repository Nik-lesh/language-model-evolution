from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

def upload_files():
    """Upload model files to HuggingFace."""
    
    # Your HuggingFace username - CHANGE THIS!
    username = input("Enter your HuggingFace username: ").strip()
    
    repo_id = f"{username}/financial-language-model"
    
    print(f"\n{'='*70}")
    print(f"📤 UPLOADING TO HUGGINGFACE")
    print(f"{'='*70}")
    print(f"\n📦 Repository: {repo_id}\n")
    
    api = HfApi()
    
    # Create repo (if doesn't exist)
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"⚠️  {e}\n")
    
    # Files to upload
    files_to_upload = [
        {
            'path': 'checkpoints/transformer_1gb_balanced_best.pth',
            'name': 'Model checkpoint',
            'repo_path': 'transformer_1gb_balanced_best.pth'
        },
        {
            'path': 'data/mega_word_dataset.pkl',
            'name': 'Word-level dataset',
            'repo_path': 'mega_word_dataset.pkl'
        },
    ]
    
    # Check files exist and show sizes
    print("📁 Files to upload:")
    for file_info in files_to_upload:
        path = Path(file_info['path'])
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"   ✓ {file_info['name']}: {size_mb:.1f} MB")
        else:
            print(f"   ✗ {file_info['name']}: NOT FOUND")
            return
    
    print(f"\n⬆️  Starting upload (this will take 10-15 minutes for 2.2 GB)...\n")
    
    # Upload files
    for file_info in files_to_upload:
        local_path = file_info['path']
        repo_path = file_info['repo_path']
        
        print(f"📤 Uploading {file_info['name']}...")
        
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"   ✅ Uploaded {repo_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return
    
    # Create README
    readme_content = f"""---
language: en
tags:
- finance
- language-model
- transformer
- financial-advisor
license: mit
---

# Financial Language Model

Custom-trained Transformer for financial text generation.

## Model Details

- **Architecture:** 6-layer Transformer
- **Parameters:** ~12M
- **Vocabulary:** 20,000 words
- **Training Data:** 1 GB balanced financial corpus (168M words)
- **Validation Loss:** 4.01
- **Modern Content:** 87%

## Training Data Composition

- Financial news (2015-2024): 750 MB (87%)
- Classical economics: 35 MB (4%)
- Wikipedia/Academic: 15 MB (2%)

## Usage
```python
from huggingface_hub import hf_hub_download
import torch
import pickle

# Download files
model_path = hf_hub_download(repo_id="{repo_id}", filename="transformer_1gb_balanced_best.pth")
dataset_path = hf_hub_download(repo_id="{repo_id}", filename="mega_word_dataset.pkl")

# Load dataset
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

# Load model
checkpoint = torch.load(model_path, map_location='cpu')
# ... create and load model ...
```

## Files

- `transformer_1gb_balanced_best.pth` - Model checkpoint (50 MB)
- `mega_word_dataset.pkl` - Preprocessed dataset (2.2 GB)

## Training Details

- Hardware: Google Colab TPU v2
- Training Time: 7.5 hours
- Epochs: 30
- Batch Size: 512
- Learning Rate: 0.0003 (adaptive)

## Project

Full project: https://github.com/{username}/language-model-evolution
"""
    
    print(f"\n📝 Creating README...")
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"   ✅ README created")
    except Exception as e:
        print(f"   ⚠️  {e}")
    
    print(f"\n{'='*70}")
    print(f"🎉 ALL DONE!")
    print(f"{'='*70}")
    print(f"\n🔗 View your model:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n📥 Download in your app:")
    print(f'''
from huggingface_hub import hf_hub_download

model = hf_hub_download(
    repo_id="{repo_id}",
    filename="transformer_1gb_balanced_best.pth"
)
''')

if __name__ == "__main__":
    upload_files()