import pdfplumber
import os
import re

def extract_text_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber."""
    print(f"Extracting text from {pdf_path}...")
    
    text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  Found {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                if page_num % 10 == 0:
                    print(f"  Processing page {page_num}/{total_pages}...")
                
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n"
            
            print(f"  ✓ Extracted {len(text):,} characters")
            return text
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return ""

def clean_extracted_text_gentle(text):
    """
    Gentle cleaning - only remove obvious junk.
    
    We'll be MUCH less aggressive this time.
    """
    print("Cleaning extracted text (gentle mode)...")
    
    original_length = len(text)
    
    # Only fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Remove ONLY excessive blank lines (4 or more)
    text = re.sub(r'\n{4,}', '\n\n', text)
    
    # Remove excessive spaces (3 or more)
    text = re.sub(r' {3,}', ' ', text)
    
    # Strip whitespace from start and end
    text = text.strip()
    
    print(f"  Original: {original_length:,} characters")
    print(f"  Cleaned: {len(text):,} characters")
    print(f"  Removed: {original_length - len(text):,} characters ({100 * (original_length - len(text)) / original_length:.1f}%)")
    
    return text

def process_pdf(pdf_path, output_filename):
    """Extract and clean PDF."""
    print("\n" + "=" * 60)
    print(f"Processing: {pdf_path}")
    print("=" * 60)
    
    # Extract
    text = extract_text_pdfplumber(pdf_path)
    
    if not text:
        print("⚠ No text extracted!")
        return ""
    
    # Clean (gently!)
    cleaned_text = clean_extracted_text_gentle(text)
    
    # Save
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"✓ Saved to: {output_path}")
    
    return cleaned_text

def process_multiple_pdfs(pdf_files, output_file='training_corpus.txt'):
    """Process multiple PDFs and combine them."""
    print("\n" + "=" * 60)
    print("PROCESSING MULTIPLE PDFs")
    print("=" * 60)
    
    all_text = ""
    
    for pdf_path, book_name in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"\n✗ File not found: {pdf_path}")
            continue
        
        text = process_pdf(pdf_path, f"{book_name}.txt")
        
        if text:
            all_text += text
            all_text += "\n\n" + "=" * 80 + "\n\n"
    
    # Save combined
    output_path = os.path.join('data', output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Combined text: {len(all_text):,} characters")
    print(f"✓ Size: {len(all_text) / 1024:.2f} KB")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    pdf_files = [
        ('pdfs/rich_dad_poor_dad.pdf', 'rich_dad'),
        ('pdfs/psychology_of_money.pdf', 'psychology_money'),
    ]
    
    process_multiple_pdfs(pdf_files)
    
    print("\n Done!")