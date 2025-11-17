import os
import zipfile
from pathlib import Path
import time
# Kaggle financial news datasets
# You need Kaggle API key: https://www.kaggle.com/docs/api
KAGGLE_DATASETS = [
    {
        'name': 'jeet2016/us-financial-news-articles',
        'description': 'US Financial News (300K+ articles, ~150 MB)',
        'size': '~150 MB'
    },
    {
        'name': 'ankurzing/sentiment-analysis-for-financial-news',
        'description': 'Financial sentiment news (~50 MB)',
        'size': '~50 MB'
    },
    {
        'name': 'notlucasp/financial-news-headlines',
        'description': 'Financial headlines and articles (~30 MB)',
        'size': '~30 MB'
    },
    {
        'name': 'miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests',
        'description': 'Stock market news database (~100 MB)',
        'size': '~100 MB'
    }
]

def setup_kaggle_api():
    """
    Setup Kaggle API credentials.
    
    1. Go to kaggle.com → Account → Create API Token
    2. Download kaggle.json
    3. Move to ~/.kaggle/kaggle.json
    """
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if not kaggle_file.exists():
        print("❌ Kaggle API not configured!")
        print("\n📝 Setup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move downloaded kaggle.json to ~/.kaggle/")
        print("5. chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API configured!")
    return True

def download_kaggle_dataset(dataset_name, output_dir='data/news/kaggle'):
    """Download and extract Kaggle dataset."""
    
    print(f"\n📥 Downloading {dataset_name}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download using Kaggle CLI
        os.system(f'kaggle datasets download -d {dataset_name} -p {output_dir} --unzip')
        
        print(f"   ✅ Downloaded and extracted")
        
        # Check what we got
        files = list(Path(output_dir).rglob('*.csv')) + list(Path(output_dir).rglob('*.txt'))
        total_size = sum(f.stat().st_size for f in files)
        
        print(f"   📦 Files: {len(files)}")
        print(f"   📏 Size: {total_size/1024/1024:.2f} MB")
        
        return True, total_size
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0

def convert_csv_to_text(csv_dir='data/news/kaggle', output_file='data/news/kaggle_combined.txt'):
    """Convert CSV articles to plain text."""
    
    import pandas as pd
    
    print("\n🔄 Converting CSV files to text...")
    
    csv_files = list(Path(csv_dir).rglob('*.csv'))
    print(f"   Found {len(csv_files)} CSV files")
    
    all_text = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Try common column names for article text
            text_columns = ['text', 'content', 'article', 'body', 'description', 'headline']
            
            for col in text_columns:
                if col in df.columns:
                    texts = df[col].dropna().astype(str).tolist()
                    all_text.extend(texts)
                    print(f"   ✅ {csv_file.name}: {len(texts)} articles from '{col}' column")
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Skipped {csv_file.name}: {e}")
            continue
    
    # Combine and save
    combined = '\n\n'.join(all_text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"\n✅ Combined text saved: {output_file}")
    print(f"   📏 Size: {len(combined)/1024/1024:.2f} MB")
    print(f"   📰 Articles: {len(all_text):,}")

def download_all_kaggle():
    """Download all Kaggle financial news datasets."""
    
    print("\n" + "="*80)
    print("📰 DOWNLOADING FINANCIAL NEWS FROM KAGGLE")
    print("="*80)
    
    if not setup_kaggle_api():
        return
    
    print(f"\n📊 Datasets to download: {len(KAGGLE_DATASETS)}\n")
    
    successful = 0
    total_size = 0
    
    for dataset in KAGGLE_DATASETS:
        print(f"\n{'='*80}")
        print(f"📦 {dataset['description']}")
        print(f"   Expected: {dataset['size']}")
        
        success, size = download_kaggle_dataset(dataset['name'])
        
        if success:
            successful += 1
            total_size += size
        
        time.sleep(5)
    
    # Convert CSVs to text
    if successful > 0:
        convert_csv_to_text()
    
    print("\n" + "="*80)
    print("📊 KAGGLE SUMMARY")
    print("="*80)
    print(f"✅ Downloaded: {successful}/{len(KAGGLE_DATASETS)} datasets")
    print(f"📦 Total: {total_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    # Install dependencies first
    print("📦 Installing dependencies...")
    os.system('pip install -q kaggle pandas')
    
    download_all_kaggle()