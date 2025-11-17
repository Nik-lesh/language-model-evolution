import praw
import time
from pathlib import Path
from datetime import datetime, timedelta

# Subreddits with quality financial discussions
FINANCE_SUBREDDITS = [
    'investing',
    'stocks',
    'personalfinance',
    'financialindependence',
    'Fire',  # Financial Independence Retire Early
    'Bogleheads',
    'StockMarket',
    'options',
    'SecurityAnalysis',
    'ValueInvesting',
    'dividends',
    'RealEstate',
    'Economics',
    'economy',
]

def download_reddit_posts(subreddit_name, limit=1000):
    """
    Download top posts from finance subreddit.
    
    Requires Reddit API credentials:
    1. Go to https://www.reddit.com/prefs/apps
    2. Create app
    3. Get client_id and client_secret
    """
    
    # You need to set these up
    reddit = praw.Reddit(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='FinanceDataCollector/1.0'
    )
    
    print(f"\n📱 r/{subreddit_name}:", end=" ")
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        posts_text = []
        
        # Get top posts from last year
        for post in subreddit.top(time_filter='year', limit=limit):
            # Combine title + selftext
            text = f"{post.title}\n\n{post.selftext}"
            
            if len(text) > 100:  # Only substantial posts
                posts_text.append(text)
        
        # Save
        output_dir = Path('data/news/reddit')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{subreddit_name}.txt'
        combined = '\n\n' + '='*80 + '\n\n'.join(posts_text)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined)
        
        chars = len(combined)
        print(f"✅ {len(posts_text)} posts, {chars/1024:.1f} KB")
        
        return True, chars
        
    except Exception as e:
        print(f"❌ {e}")
        return False, 0

def download_all_reddit():
    """Download from all finance subreddits."""
    
    print("\n" + "="*80)
    print("💬 REDDIT FINANCE COMMUNITIES")
    print("="*80)
    print("\n⚠️  Requires Reddit API credentials")
    print("   Setup: https://www.reddit.com/prefs/apps")
    print(f"\n📊 Subreddits: {len(FINANCE_SUBREDDITS)}")
    print(f"📄 Posts per sub: 1000")
    print(f"📦 Expected: 50-100 MB\n")
    
    total_chars = 0
    successful = 0
    
    for subreddit in FINANCE_SUBREDDITS:
        success, chars = download_reddit_posts(subreddit, limit=1000)
        
        if success:
            successful += 1
            total_chars += chars
        
        time.sleep(3)
    
    print("\n" + "="*80)
    print("📊 REDDIT SUMMARY")
    print("="*80)
    print(f"✅ Success: {successful}/{len(FINANCE_SUBREDDITS)}")
    print(f"📦 Total: {total_chars/1024/1024:.2f} MB")

if __name__ == "__main__":
    import os
    os.system('pip install -q praw')
    
    download_all_reddit()