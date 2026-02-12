from news.fetch_news import fetch_news
from inference.predict import predict
from collections import Counter


def analyze_stock_news(stock: str, limit: int = 100):
    """
    Fetch news for a stock, run sentiment prediction,
    and return:
        1) summary dict
        2) list of article dicts
    """

    news_articles = fetch_news(stock, limit)

    sentiments = []
    detailed_results = []

    for article in news_articles:

        # If fetch_news returns dict → use title
        if isinstance(article, dict):
            title = article.get("title", "")
        else:
            # If fetch_news returns string → use directly
            title = str(article)

        if not title:
            continue

        label = predict(title)

        sentiments.append(label)

        detailed_results.append({
            "title": title,
            "sentiment": label
        })

    counter = Counter(sentiments)

    buy_count = counter.get("BUY", 0)
    sell_count = counter.get("SELL", 0)
    hold_count = counter.get("HOLD", 0)

    score = buy_count - sell_count

    if score >= 5:
        signal = "🚨 STRONG BUY"
    elif score <= -5:
        signal = "🚨 STRONG SELL"
    elif score > 0:
        signal = "Bullish"
    elif score < 0:
        signal = "Bearish"
    else:
        signal = "Neutral"

    summary = {
        "stock": stock,
        "buy": buy_count,
        "sell": sell_count,
        "hold": hold_count,
        "signal": signal
    }

    return summary, detailed_results