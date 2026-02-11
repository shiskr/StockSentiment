import requests

API_KEY = "TOKEN"

def fetch_news(ticker, limit=10):
    url = "https://api.polygon.io/v2/reference/news"

    params = {
        "ticker": ticker,
        "limit": limit,
        "apiKey": API_KEY
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    articles = []

    for item in data.get("results", []):
        title = item.get("title", "")
        description = item.get("description", "")

        # Combine title + description
        full_text = f"{title}. {description}"
        articles.append(full_text.strip())

    return articles