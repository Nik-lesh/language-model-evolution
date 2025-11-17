from pathlib import Path
import re

def clean_text(text):
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def combine_all_sources():
    """Combine ALL sources into balanced corpus."""
    
    print("\n" + "="*80)
    print("🎯 CREATING BALANCED FINAL CORPUS")
    print("="*80)
    
    sources = {
        'Modern News (Kaggle)': [
            'data/news/kaggle_financial_news.txt',
            'data/news/kaggle_financial_news2.txt',
            'data/news/kaggle_financial_news3.txt',
            'data/news/kaggle_financial_news4.txt',
            'data/news/kaggle_financial_news5.txt',
        ],
        'Financial Phrasebank (HF)': [
            'data/news/Sentences_50Agree.txt',
            'data/news/Sentences_66Agree.txt',
            'data/news/Sentences_75Agree.txt',
            'data/news/Sentences_AllAgree.txt',
        ],
        'Twitter Financial (HF)': [
            'data/news/twitter_financial.txt',
        ],
        'Financial Classification': [
            'data/news/financial_classification.txt',
        ],
        'Classical Economics (Original)': [
            'data//news/mega_corpus.txt',  
        ],
        "Stock News CSV Text": [
            'data/news/stock_market_news.txt',
        ],
    }
    
    combined_text = ""
    stats = {}
    
    for source_name, file_list in sources.items():
        print(f"\n📚 Processing {source_name}...")
        
        source_text = ""
        files_found = 0
        
        for filepath in file_list:
            file_path = Path(filepath)
            
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # For classical corpus, only take 30% (balance it!)
                if 'Classical' in source_name:
                    # Sample every 3rd chunk to reduce from 103MB to ~30MB
                    chunks = text.split('\n\n')
                    text = '\n\n'.join(chunks[::3])  # Take every 3rd
                    print(f"   📊 Sampling classical corpus to 30%...")
                
                text = clean_text(text)
                
                if len(text) > 1000:
                    source_text += text + "\n\n" + "="*80 + "\n\n"
                    files_found += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error reading {file_path.name}: {e}")
                continue
        
        combined_text += source_text
        
        stats[source_name] = {
            'files': files_found,
            'chars': len(source_text),
            'mb': len(source_text) / 1024 / 1024
        }
        
        if files_found > 0:
            print(f"   ✅ {files_found} files, {stats[source_name]['mb']:.1f} MB")
        else:
            print(f"   ⚠️ No files found")
    
    # Save balanced corpus
    output_file = 'data/balanced_corpus.txt'
    
    print(f"\n💾 Saving balanced corpus...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    # Statistics
    total_chars = len(combined_text)
    total_mb = total_chars / 1024 / 1024
    words = combined_text.split()
    
    print("\n" + "="*80)
    print("🎊 BALANCED CORPUS CREATED!")
    print("="*80)
    
    print("\n📊 SOURCE BREAKDOWN:")
    for source, stat in stats.items():
        if stat['mb'] > 0:
            pct = 100 * stat['mb'] / total_mb
            print(f"   {source:30} {stat['files']:3} files  {stat['mb']:7.1f} MB ({pct:5.1f}%)")
    
    print(f"\n📏 TOTAL STATISTICS:")
    print(f"   Size: {total_chars:,} characters ({total_mb:.2f} MB)")
    print(f"   Words: {len(words):,}")
    print(f"   Avg word length: {total_chars/len(words):.2f} chars")
    
    # Era breakdown
    modern_mb = sum(s['mb'] for name, s in stats.items() 
                   if 'Modern' in name or 'Twitter' in name or 'Phrasebank' in name or 'Classification' in name)
    classical_mb = sum(s['mb'] for name, s in stats.items() if 'Classical' in name)
    
    print(f"\n📅 ERA BALANCE:")
    print(f"   Modern (2010-2024):    {modern_mb:6.1f} MB ({100*modern_mb/total_mb:5.1f}%)")
    print(f"   Classical (pre-1900):  {classical_mb:6.1f} MB ({100*classical_mb/total_mb:5.1f}%)")
    
    if modern_mb > classical_mb:
        print(f"\n   ✅ BALANCED! Modern content dominates ({modern_mb/classical_mb:.1f}x)")
    else:
        print(f"\n   ⚠️ Still classical-heavy (need more modern)")
    
    # Save metadata
    import json
    metadata = {
        'total_mb': total_mb,
        'total_words': len(words),
        'sources': stats,
        'modern_pct': 100 * modern_mb / total_mb,
        'classical_pct': 100 * classical_mb / total_mb
    }
    
    with open('data/balanced_corpus_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📋 Metadata saved to: data/balanced_corpus_metadata.json")
    print("="*80)

if __name__ == "__main__":
    combine_all_sources()