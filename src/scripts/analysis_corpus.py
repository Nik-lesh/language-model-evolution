import re
from collections import Counter

def analyze_text(filepath):
    """Analyze our training corpus."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("=" * 60)
    print("CORPUS ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    print(f"\nTotal characters: {len(text):,}")
    print(f"Total size: {len(text) / 1024:.2f} KB")
    
    # Character level stats
    unique_chars = sorted(set(text))
    print(f"\nUnique characters: {len(unique_chars)}")
    print(f"Character set preview: {''.join(unique_chars[:50])}...")
    
    # Word level stats
    words = re.findall(r'\b\w+\b', text.lower())
    print(f"\nTotal words: {len(words):,}")
    print(f"Unique words: {len(set(words)):,}")
    
    # Most common words
    word_counts = Counter(words)
    print("\nTop 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word:15s} {count:,}")
    
    # Financial terms
    financial_terms = [
        'money', 'invest', 'wealth', 'asset', 'income',
        'financial', 'rich', 'poor', 'business', 'cash'
    ]
    
    print("\nFinancial term frequency:")
    for term in financial_terms:
        count = len(re.findall(rf'\b{term}\w*\b', text.lower()))
        print(f"  {term:12s} {count:,}")
    
    # Average word length
    avg_word_len = sum(len(w) for w in words) / len(words)
    print(f"\nAverage word length: {avg_word_len:.2f} characters")
    
    # Sentences (rough estimate)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Approximate sentences: {len(sentences):,}")
    
    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE TEXT (characters 10000-10500):")
    print("=" * 60)
    print(text[10000:10500])
    print("...")

if __name__ == "__main__":
    # Analyze the cleaned corpus
    analyze_text('data/training_corpus_clean.txt')