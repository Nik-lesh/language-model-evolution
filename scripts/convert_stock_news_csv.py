import pandas as pd
from pathlib import Path

def convert_stock_news_csv():
    """Convert stock market news CSV to text."""
    
    print("\n" + "="*80)
    print("📊 CONVERTING STOCK NEWS CSV TO TEXT")
    print("="*80)
    
    # Find CSV files
    csv_dir = Path('data/news')
    csv_files = list(csv_dir.glob('*.csv'))
    
    print(f"\n📁 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   {f.name}: {f.stat().st_size/1024/1024:.1f} MB")
    
    all_articles = []
    
    for csv_file in csv_files:
        print(f"\n📄 Processing {csv_file.name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            
            # Try to find text columns
            # Common names: text, content, article, body, headline, title, description
            text_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['text', 'content', 'article', 'body', 'headline', 'news', 'title']):
                    text_columns.append(col)
            
            print(f"   Text columns found: {text_columns}")
            
            # Extract text from all text columns
            for col in text_columns:
                texts = df[col].dropna().astype(str).tolist()
                
                # Filter out very short texts
                texts = [t for t in texts if len(t) > 50]
                
                all_articles.extend(texts)
                print(f"   ✅ Extracted {len(texts):,} texts from '{col}'")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Combine all articles
    print(f"\n📝 Combining {len(all_articles):,} articles...")
    
    combined = '\n\n'.join(all_articles)
    
    # Save
    output_file = 'data/news/stock_market_news.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    # Statistics
    total_chars = len(combined)
    total_words = len(combined.split())
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"📦 Output: {output_file}")
    print(f"📏 Size: {total_chars:,} characters ({total_chars/1024/1024:.2f} MB)")
    print(f"📝 Words: {total_words:,}")
    print(f"📰 Articles: {len(all_articles):,}")
    print(f"📖 Average article: {total_chars/len(all_articles):.0f} characters")
    
    # Show sample
    print("\n" + "="*80)
    print("📄 SAMPLE (First 300 chars)")
    print("="*80)
    print(all_articles[0][:300] + "...")

if __name__ == "__main__":
    convert_stock_news_csv()