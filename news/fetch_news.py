import requests

API_KEY = "TOKEN"
API_URL = "https://newsapi.org/v2/top-headlines?"

def fetch_news(ticker, limit=10):
    url = API_URL

    params = {
        "ticker": ticker,
        "limit": limit,
        "apiKey": API_KEY
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    return [item["title"] for item in data.get("results", [])]