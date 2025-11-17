import requests
import time
from pathlib import Path
import json

# Modern finance books from Internet Archive (2000-2024)
# These are borrowable digital books (legal!)
MODERN_FINANCE_BOOKS = [
    # Investment Classics (2000s-2020s)
    'intelligentinvestor00benj',  # Benjamin Graham
    'randomwalkdownwal00malk',  # Burton Malkiel
    'commonst0000fish',  # Philip Fisher
    'oneuponstree00lync',  # Peter Lynch
    'littlebookofcomm00bogl',  # John Bogle
    'littlebookthatbe00gree',  # Joel Greenblatt
    'essayofwarrenbu00buff',  # Warren Buffett
    
    # Personal Finance (Modern)
    'millionairenextd00stan',  # Thomas Stanley
    'yourmoneyyourli00robi',  # Vicki Robin
    'iwillteachyout00seti',  # Ramit Sethi
    'totalmoneymakeo00rams',  # Dave Ramsey
    'richestmaninbab00clas',  # George Clason (classic but readable)
    'automaticmillion00bach',  # David Bach
    'richbrokenorth00eker',  # T. Harv Eker
    
    # Modern Economics/Behavioral
    'freakonomics00levi',  # Steven Levitt
    'thinkingfastand00kahn',  # Daniel Kahneman
    'nudgeimprovingd00thal',  # Richard Thaler
    'predictablyirra00arie',  # Dan Ariely
    'blackswanimpact00tale',  # Nassim Taleb
    'misbehavingmaki00thal',  # Thaler
    
    # Business/Investing Stories (Narrative, Modern)
    'bigshortinside00lewi',  # Michael Lewis
    'liarspokerrisin00lewi',  # Michael Lewis
    'flashboyswallst00lewi',  # Michael Lewis
    'barbariansatthe00burr',  # Bryan Burrough
    'whengeniosityfa00lown',  # Roger Lowenstein
    
    # Investment Strategy (Modern)
    'commonsensinves00swem',  # John Bogle
    'fourhourworkweek00ferr',  # Tim Ferriss (entrepreneurship)
    'zerotoonepeter00thie',  # Peter Thiel
    'leanstartuphowe00ries',  # Eric Ries
    
    # Trading/Markets (Contemporary)
    'marketwizardsint00schw',  # Jack Schwager
    'reminiscencesof00lefe',  # Jesse Livermore
    'waytowealthblog00hill',  # Napoleon Hill
]

def search_archive_org(query, max_results=50):
    """
    Search Internet Archive for finance books.
    
    Returns identifiers for books matching query.
    """
    
    search_url = "https://archive.org/advancedsearch.php"
    
    params = {
        'q': query,
        'fl[]': ['identifier', 'title', 'creator', 'year'],
        'sort[]': 'downloads desc',
        'rows': max_results,
        'page': 1,
        'output': 'json'
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data['response']['docs']
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []

def download_from_archive(identifier, output_dir='data/books/archive_modern'):
    """
    Download text from Internet Archive.
    
    Archive.org provides multiple formats - we'll try to get plain text.
    """
    
    print(f"📥 Downloading {identifier}...", end=" ")
    
    # Try to get plain text version
    text_url = f"https://archive.org/stream/{identifier}/{identifier}_djvu.txt"
    
    try:
        response = requests.get(text_url, timeout=60)
        
        if response.status_code == 200 and len(response.text) > 10000:
            # Save
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(output_dir) / f"{identifier}.txt"
            
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(response.text)
            
            size_kb = len(response.text) / 1024
            print(f"✅ {size_kb:.1f} KB")
            return True, len(response.text)
        else:
            print(f"❌ No text version")
            return False, 0
            
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return False, 0

def download_all_modern_books():
    """Download modern finance books from Archive.org."""
    
    print("\n" + "="*80)
    print("📚 DOWNLOADING MODERN FINANCE BOOKS FROM INTERNET ARCHIVE")
    print("="*80)
    print(f"\n📊 Target books: {len(MODERN_FINANCE_BOOKS)}")
    print("📜 Source: archive.org (legal borrowing)")
    print("⏱️  Estimated time: 20-30 minutes\n")
    
    successful = []
    failed = []
    total_chars = 0
    
    for i, identifier in enumerate(MODERN_FINANCE_BOOKS, 1):
        print(f"[{i}/{len(MODERN_FINANCE_BOOKS)}] ", end="")
        
        success, chars = download_from_archive(identifier)
        
        if success:
            successful.append(identifier)
            total_chars += chars
        else:
            failed.append(identifier)
        
        time.sleep(3)  # Be respectful
    
    # Summary
    print("\n" + "="*80)
    print("📊 ARCHIVE.ORG DOWNLOAD SUMMARY")
    print("="*80)
    print(f"✅ Successful: {len(successful)}/{len(MODERN_FINANCE_BOOKS)}")
    print(f"❌ Failed: {len(failed)}")
    
    if total_chars > 0:
        print(f"📦 Total: {total_chars/1024/1024:.2f} MB")
        print(f"📖 Average: {total_chars/len(successful)/1024:.1f} KB per book")
    
    # Save successful IDs for reference
    with open('data/books/archive_modern/downloaded.json', 'w') as f:
        json.dump({
            'successful': successful,
            'failed': failed,
            'total_chars': total_chars
        }, f, indent=2)
    
    if failed:
        print(f"\n💡 {len(failed)} books unavailable in text format")
        print("   Can try manual download or skip")

if __name__ == "__main__":
    download_all_modern_books()