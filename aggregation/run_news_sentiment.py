from news.fetch_news import fetch_news
from inference.predict import predict
from collections import Counter
from storage.db import save_result, init_db

init_db()


def analyze_stock_news(ticker, limit=100):
    articles = fetch_news(ticker, limit=limit)

    # Deduplicate based on title/text
    seen = set()
    unique_articles = []

    for article in articles:
        title = article.strip()
        if title not in seen:
            seen.add(title)
            unique_articles.append(title)

    articles = unique_articles

    results = []
    detailed_results = []

    for article in articles:
        label, probs = predict(article)
        results.append(label)
        clean_title = article.strip().split(".")[0]  # first sentence only
        detailed_results.append((clean_title, label))

        print("\n--- ARTICLE ---")
        print(article[:200], "...")
        print("Sentiment:", label)

    summary = Counter(results)

    print("\n=== SUMMARY ===")
    print(summary)
    buy = summary["BUY"]
    sell = summary["SELL"]
    hold = summary["HOLD"]
    save_result(ticker, buy, sell, hold)

    score = buy - sell

    print("\nSentiment Score:", score)

    if score >= 5:
        print("🚨 STRONG BUY SIGNAL 🚀")
    elif score <= -5:
        print("🚨 STRONG SELL SIGNAL 🔻")
    elif buy > sell:
        print("Overall sentiment: BULLISH 📈")
    elif sell > buy:
        print("Overall sentiment: BEARISH 📉")
    else:
        print("Overall sentiment: NEUTRAL ⚖️")
    return detailed_results

