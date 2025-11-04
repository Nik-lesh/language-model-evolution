import re

def remove_front_matter(text):
    """
    Remove copyright, front matter, table of contents.
    Keep only the actual book content.
    """
    print("Removing front matter and legal text...")
    
    original_length = len(text)
    
    # Find where the actual content starts
    # Look for common chapter markers
    chapter_markers = [
        "Lesson 1:",
        "LESSON 1:",
        "Chapter 1",
        "CHAPTER 1",
        "Introduction",
        "INTRODUCTION"
    ]
    
    # Try to find first chapter
    start_pos = 0
    for marker in chapter_markers:
        pos = text.find(marker)
        if pos != -1 and (start_pos == 0 or pos < start_pos):
            # Found a chapter marker earlier in the text
            # Go back a bit to include any intro paragraphs
            start_pos = max(0, pos - 500)
    
    if start_pos > 0:
        text = text[start_pos:]
        print(f"  Removed {original_length - len(text):,} characters of front matter")
    
    # Remove common copyright phrases
    copyright_patterns = [
        r'Copyright © \d{4}.*?(?=\n\n)',
        r'All rights reserved\..*?(?=\n\n)',
        r'ISBN:.*?(?=\n)',
        r'Published by.*?(?=\n\n)',
    ]
    
    for pattern in copyright_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    print(f"  Final length: {len(text):,} characters")
    
    return text.strip()

def main():
    # Read the corpus
    with open('data/training_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original corpus: {len(text):,} characters")
    
    # Clean it
    cleaned = remove_front_matter(text)
    
    # Save cleaned version
    with open('data/training_corpus_clean.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"\n✓ Saved cleaned corpus to: data/training_corpus_clean.txt")
    print(f"  Size: {len(cleaned) / 1024:.2f} KB")
    
    # Show first 500 characters
    print("\nFirst 500 characters of cleaned text:")
    print("=" * 60)
    print(cleaned[:500])
    print("...")

if __name__ == "__main__":
    main()