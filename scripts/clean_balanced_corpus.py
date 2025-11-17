import re
from pathlib import Path
from collections import Counter
import unicodedata

def clean_text_comprehensive(text):
    """
    Comprehensive text cleaning for ML training.
    
    Removes noise while preserving financial terminology.
    """
    
    # 1. Fix unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Fix common escaped characters
    replacements = {
        '\\n': '\n',
        '\\t': ' ',
        '\\r': '',
        '\\"': '"',
        "\\'": "'",
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&nbsp;': ' ',
        '\\u2019': "'",
        '\\u201c': '"',
        '\\u201d': '"',
        '\\u2014': '—',
        '\\u2013': '–',
        '\\xa0': ' ',
        '\\u2026': '...',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 4. Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # 5. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 6. Remove common web junk
    junk_patterns = [
        r'Click here to .*',
        r'Subscribe to .*',
        r'Follow us on .*',
        r'Share this article.*',
        r'Read more:.*',
        r'Related Articles:.*',
        r'Advertisement',
        r'ADVERTISEMENT',
        r'Copyright © \d{4}.*',
        r'All rights reserved.*',
        r'\[Advertisement\]',
        r'Terms of Service.*',
        r'Privacy Policy.*',
    ]
    
    for pattern in junk_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 7. Fix excessive punctuation
    text = re.sub(r'\.{4,}', '...', text)  # Multiple dots
    text = re.sub(r'!{2,}', '!', text)     # Multiple exclamations
    text = re.sub(r'\?{2,}', '?', text)    # Multiple questions
    
    # 8. Fix whitespace
    text = re.sub(r' {2,}', ' ', text)      # Multiple spaces
    text = re.sub(r'\n{4,}', '\n\n', text)  # Excessive newlines
    text = re.sub(r'\t+', ' ', text)        # Tabs to spaces
    
    # 9. Remove lines that are just numbers or special chars
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip very short lines
        if len(line) < 10:
            continue
        
        # Skip lines that are mostly numbers/special chars
        alpha_chars = sum(c.isalpha() for c in line)
        if alpha_chars / len(line) < 0.5:  # At least 50% letters
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 10. Remove non-ASCII characters (keep standard punctuation)
    text = ''.join(char for char in text if ord(char) < 128 or char in '—–""''')
    
    # 11. Final whitespace cleanup
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def remove_duplicates(text, chunk_size=200):
    """
    Remove duplicate chunks of text.
    
    Common in scraped data where same article appears multiple times.
    """
    
    print("\n🔍 Removing duplicates...")
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Track seen chunks
    seen = set()
    unique_sentences = []
    duplicates_removed = 0
    
    for sentence in sentences:
        # Create fingerprint (first 100 chars)
        fingerprint = sentence[:100].lower()
        
        if fingerprint not in seen and len(fingerprint) > 20:
            seen.add(fingerprint)
            unique_sentences.append(sentence)
        else:
            duplicates_removed += 1
    
    text = '. '.join(unique_sentences)
    
    print(f"   Removed {duplicates_removed:,} duplicate sentences")
    print(f"   Kept {len(unique_sentences):,} unique sentences")
    
    return text

def clean_corpus(input_file='data/balanced_corpus.txt', 
                 output_file='data/balanced_corpus_clean.txt'):
    """Clean the entire balanced corpus."""
    
    print("\n" + "="*80)
    print("🧹 CLEANING BALANCED CORPUS")
    print("="*80)
    
    print(f"\n📂 Input: {input_file}")
    
    # Read
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    original_size = len(text)
    original_words = len(text.split())
    
    print(f"   Original: {original_size:,} chars ({original_size/1024/1024:.2f} MB)")
    print(f"   Words: {original_words:,}")
    
    # Clean
    print("\n🔧 Cleaning...")
    print("   1. Fixing unicode and HTML...")
    text = clean_text_comprehensive(text)
    
    print("   2. Removing duplicates...")
    text = remove_duplicates(text)
    
    print("   3. Final validation...")
    
    # Remove empty paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    text = '\n\n'.join(paragraphs)
    
    # Save
    print(f"\n💾 Saving cleaned corpus...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Statistics
    clean_size = len(text)
    clean_words = len(text.split())
    removed_size = original_size - clean_size
    removed_pct = 100 * removed_size / original_size
    
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE!")
    print("="*80)
    print(f"\n📊 BEFORE:")
    print(f"   Size: {original_size:,} chars ({original_size/1024/1024:.2f} MB)")
    print(f"   Words: {original_words:,}")
    
    print(f"\n📊 AFTER:")
    print(f"   Size: {clean_size:,} chars ({clean_size/1024/1024:.2f} MB)")
    print(f"   Words: {clean_words:,}")
    
    print(f"\n🗑️ REMOVED:")
    print(f"   Size: {removed_size:,} chars ({removed_size/1024/1024:.2f} MB)")
    print(f"   Percentage: {removed_pct:.1f}%")
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Sample
    print("\n" + "="*80)
    print("📄 SAMPLE CLEANED TEXT")
    print("="*80)
    print(text[:500])
    print("...")

if __name__ == "__main__":
    clean_corpus()