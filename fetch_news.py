import requests
import pandas as pd
from datetime import datetime
import os
from rapidfuzz import fuzz

# -----------------------------
# 1. API keys and endpoints
# -----------------------------
NEWSAPI_KEY = "722fa8d4ef774721886e22d78e5ef8cd"
CURRENTSAPI_KEY = "ZFScZ-YrlrN74NACLcEYE7k05x6AXnfrTEoz8uwgmFCgimf0"
MEDIASTACK_KEY = "12b2a5c0f24122d7e28d364a44c7dc87"

NEWSAPI_URL = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWSAPI_KEY}"
CURRENTSAPI_URL = f"https://api.currentsapi.services/v1/latest-news?language=en&apiKey={CURRENTSAPI_KEY}"
MEDIASTACK_URL = f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_KEY}&countries=in&languages=en&limit=100"

# -----------------------------
# 2. Collect news from APIs
# -----------------------------
all_news = []

def add_articles(articles, source_name, content_key='content'):
    for article in articles:
        all_news.append({
            'title': article.get('title'),
            'description': article.get('description') or "",
            'content': article.get(content_key) or article.get('description') or "",
            'source': source_name,
            'published_at': article.get('publishedAt') or article.get('published') or "",
            'label': 1  # True news
        })

def fetch_api(url, source_name, content_key='content'):
    try:
        response = requests.get(url, timeout=10)  # <- timeout added
        data = response.json()
        if source_name == "NewsAPI":
            articles = data.get('articles', [])
        elif source_name == "CurrentsAPI":
            articles = data.get('news', [])
        elif source_name == "MediaStack":
            articles = data.get('data', [])
        else:
            articles = []
        add_articles(articles, source_name, content_key=content_key)
        print(f"{source_name} articles fetched: {len(articles)}")
    except Exception as e:
        print(f"{source_name} error: {e}")

# Fetch from APIs
fetch_api(NEWSAPI_URL, "NewsAPI", content_key='content')
fetch_api(CURRENTSAPI_URL, "CurrentsAPI", content_key='description')
fetch_api(MEDIASTACK_URL, "MediaStack", content_key='description')

# -----------------------------
# 3. Convert to DataFrame
# -----------------------------
df_new = pd.DataFrame(all_news)
df_new['published_at'] = pd.to_datetime(df_new['published_at'], errors='coerce')
df_new['text_combined'] = df_new['title'].fillna('') + " " + df_new['content'].fillna('')

if df_new.empty:
    print("No news fetched. Exiting.")
    exit()

# -----------------------------
# 4. Append to main_data.csv with fuzzy duplicate removal
# -----------------------------
file_name = "main_data.csv"

if os.path.exists(file_name):
    df_existing = pd.read_csv(file_name)
    df_existing['published_at'] = pd.to_datetime(df_existing['published_at'], errors='coerce')

    # Combine
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Fuzzy duplicate removal
    unique_indices = []
    titles = df_combined['title'].fillna("").tolist()
    for i, title in enumerate(titles):
        is_duplicate = False
        for j in unique_indices:
            if fuzz.token_set_ratio(title, titles[j]) > 90:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
    df_final = df_combined.iloc[unique_indices].reset_index(drop=True)
else:
    df_final = df_new

# Save CSV
df_final.to_csv(file_name, index=False)
print(f"CSV saved: {file_name}. Total articles: {len(df_final)}")
