import json
from pathlib import Path

def extract_text_from_json(json_file):
    """
    Extract text from Kaggle financial news JSON.
    
    Each line is a separate JSON object with article data.
    We want the "text" field from each.
    """
    
    print(f"📄 Processing {json_file.name}...")
    
    articles = []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Each line is a JSON object
                    data = json.loads(line.strip())
                    
                    # Extract text field
                    if 'text' in data and data['text']:
                        text = data['text'].strip()
                        
                        # Only include substantial articles (>100 chars)
                        if len(text) > 100:
                            articles.append(text)
                    
                    # Progress update
                    if line_num % 10000 == 0:
                        print(f"   Processed {line_num:,} lines, {len(articles):,} articles...")
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"   ✅ Extracted {len(articles):,} articles")
        return articles
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

def process_all_kaggle_json():
    """Process all Kaggle JSON files."""
    
    print("\n" + "="*80)
    print("📰 CONVERTING KAGGLE JSON TO TEXT")
    print("="*80)
    
    # Find all JSON files
    kaggle_dir = Path('data/news/kaggle/archive/2018_05_112b52537b67659ad3609a234388c50a')

    
    if not kaggle_dir.exists():
        print(f"\n❌ Directory not found: {kaggle_dir}")
        print("\n💡 Create it and put your unzipped Kaggle files there:")
        print(f"   mkdir -p {kaggle_dir}")
        print(f"   mv ~/Downloads/us-financial-news-articles/* {kaggle_dir}/")
        return
    
    json_files = list(kaggle_dir.glob('*.json'))
    
    print(f"\n📊 Found {len(json_files)} JSON files\n")
    
    if not json_files:
        print("❌ No JSON files found!")
        print(f"\n📁 Files in {kaggle_dir}:")
        for f in kaggle_dir.iterdir():
            print(f"   {f.name}")
        return
    
    all_articles = []
    
    for json_file in json_files:
        articles = extract_text_from_json(json_file)
        all_articles.extend(articles)
    
    # Combine all articles
    print(f"\n📝 Combining {len(all_articles):,} articles...")
    
    combined_text = '\n\n'.join(all_articles)
    
    # Save
    output_file = 'data/news/kaggle_financial_news5.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    # Statistics
    total_chars = len(combined_text)
    total_words = len(combined_text.split())
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"📦 Output: {output_file}")
    print(f"📏 Size: {total_chars:,} characters ({total_chars/1024/1024:.2f} MB)")
    print(f"📝 Words: {total_words:,}")
    print(f"📰 Articles: {len(all_articles):,}")
    print(f"📖 Average article: {total_chars/len(all_articles):.0f} characters")
    
    # Sample
    print("\n" + "="*80)
    print("📄 SAMPLE ARTICLE")
    print("="*80)
    print(all_articles[0][:500] + "...")

if __name__ == "__main__":
    process_all_kaggle_json()