from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

def upload_model_to_hf():
    """Upload model and dataset to HuggingFace."""
    
    print("="*70)
    print("📤 UPLOADING TO HUGGINGFACE")
    print("="*70)
    
    # Your HuggingFace username
    username = "Nikhilesh9"
    repo_id = f"{username}/financial-language-model"
    
    api = HfApi()
    
    # Files to upload
    files = {
        'model': 'checkpoints/transformer_1gb_balanced_best.pth',
        'dataset': 'data/mega_word_dataset.pkl',
        'config': 'backend/config/settings.py',
    }
    
    print(f"\n📦 Repository: {repo_id}")
    print(f"\n📁 Files to upload:")
    
    for name, path in files.items():
        file_path = Path(path)
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"   ✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {name}: {path} (NOT FOUND)")
    
    # Upload each file
    print(f"\n⬆️  Uploading files...")
    
    for name, path in files.items():
        file_path = Path(path)
        
        if not file_path.exists():
            print(f"   ⚠️  Skipping {name} (not found)")
            continue
        
        print(f"\n📤 Uploading {name}...")
        print(f"   File: {path}")
        
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"   ✅ Uploaded!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"\n🔗 View at: https://huggingface.co/{repo_id}")
    print(f"\n📝 Files available:")
    print(f"   - transformer_1gb_balanced_best.pth")
    print(f"   - mega_word_dataset.pkl")

if __name__ == "__main__":
    upload_model_to_hf()