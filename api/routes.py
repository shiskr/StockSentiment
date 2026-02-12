from flask import render_template, request
from aggregation.run_news_sentiment import analyze_stock_news

# In-memory storage (NO session usage)
LAST_RESULTS = []               # list of stock summaries (newest first)
LAST_ANALYZED_ARTICLES = {}     # dict: { ticker: [ {title, sentiment}, ... ] }
TICKER_HISTORY_ORDER = []      # maintains ticker display order (newest first)


def register_routes(app):

    @app.route("/", methods=["GET", "POST"])
    def dashboard():
        global LAST_RESULTS, LAST_ANALYZED_ARTICLES, TICKER_HISTORY_ORDER

        if request.method == "POST":

            ticker_input = request.form.get("ticker", "")
            article_limit = request.form.get("limit", 100)

            # Safe int conversion
            try:
                article_limit = int(article_limit)
            except ValueError:
                article_limit = 100

            # Support multi-stock input
            tickers = [
                t.strip().upper()
                for t in ticker_input.split(",")
                if t.strip()
            ]

            for ticker in tickers:
                summary, articles = analyze_stock_news(
                    ticker,
                    limit=article_limit
                )

                # Remove old summary for same ticker
                LAST_RESULTS = [
                    r for r in LAST_RESULTS
                    if r.get("stock") != ticker
                ]

                # Remove ticker from history order if already present
                if ticker in TICKER_HISTORY_ORDER:
                    TICKER_HISTORY_ORDER.remove(ticker)

                # Insert newest summary at beginning
                LAST_RESULTS.insert(0, summary)

                # Insert ticker at beginning of history order
                TICKER_HISTORY_ORDER.insert(0, ticker)

                # Update latest articles for ticker
                LAST_ANALYZED_ARTICLES[ticker] = articles

        # Reorder articles dict based on ticker history
        ordered_articles = {
            ticker: LAST_ANALYZED_ARTICLES[ticker]
            for ticker in TICKER_HISTORY_ORDER
        }

        return render_template(
            "dashboard.html",
            results=LAST_RESULTS,
            analyzed_articles=ordered_articles
        )