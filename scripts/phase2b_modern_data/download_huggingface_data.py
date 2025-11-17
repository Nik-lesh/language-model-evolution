from datasets import load_dataset
from pathlib import Path
import time

def download_hf_dataset(dataset_name, config=None, split='train'):
    """Download dataset from Hugging Face."""
    
    print(f"\n📥 Downloading {dataset_name}...")
    
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        return dataset
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def save_dataset_to_text(dataset, output_file, text_column='text'):
    """Save HuggingFace dataset to text file."""
    
    print(f"   💾 Saving to {output_file}...")
    
    all_text = []
    
    # Try different common column names
    possible_columns = [text_column, 'sentence', 'content', 'article', 'headline', 'text']
    
    column_found = None
    for col in possible_columns:
        if col in dataset.column_names:
            column_found = col
            break
    
    if not column_found:
        print(f"   ❌ No text column found. Columns: {dataset.column_names}")
        return 0
    
    print(f"   📝 Using column: '{column_found}'")
    
    for example in dataset:
        text = str(example[column_found])
        if len(text) > 50:
            all_text.append(text)
    
    # Save
    combined = '\n\n'.join(all_text)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    chars = len(combined)
    print(f"   ✅ Saved {len(all_text):,} examples, {chars/1024/1024:.2f} MB")
    
    return chars

def download_all_hf_datasets():
    """Download multiple financial datasets from Hugging Face."""
    
    print("\n" + "="*80)
    print("🤗 DOWNLOADING FROM HUGGING FACE")
    print("="*80)
    
    # Financial datasets on Hugging Face (all open-access!)
    datasets_to_download = [
        {
            'name': 'financial_phrasebank',
            'config': 'sentences_allagree',
            'output': 'data/news/hf/financial_phrasebank.txt',
            'description': 'Financial sentiment sentences',
            'size': '~2 MB'
        },
        {
            'name': 'zeroshot/twitter-financial-news-sentiment',
            'config': None,
            'output': 'data/news/hf/twitter_financial.txt',
            'description': 'Twitter financial news',
            'size': '~5 MB'
        },
        {
            'name': 'nickmuchi/financial-classification',
            'config': None,
            'output': 'data/news/hf/financial_classification.txt',
            'description': 'Financial text classification',
            'size': '~3 MB'
        },
        {
            'name': 'ought/raft',
            'config': 'banking_77',
            'output': 'data/news/hf/banking_intent.txt',
            'description': 'Banking customer intents',
            'size': '~1 MB'
        },
    ]
    
    total_chars = 0
    successful = 0
    
    for ds_info in datasets_to_download:
        print(f"\n{'='*80}")
        print(f"📦 {ds_info['description']} ({ds_info['size']})")
        
        dataset = download_hf_dataset(ds_info['name'], ds_info['config'])
        
        if dataset:
            chars = save_dataset_to_text(dataset, ds_info['output'])
            if chars > 0:
                successful += 1
                total_chars += chars
        
        time.sleep(1)
    
    print("\n" + "="*80)
    print("📊 HUGGING FACE SUMMARY")
    print("="*80)
    print(f"✅ Success: {successful}/{len(datasets_to_download)}")
    print(f"📦 Total: {total_chars/1024/1024:.2f} MB")
    
    return total_chars

if __name__ == "__main__":
    import os
    
    print("📦 Installing Hugging Face datasets...")
    os.system('pip install -q datasets')
    
    download_all_hf_datasets()