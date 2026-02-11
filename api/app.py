import sqlite3
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
from aggregation.run_news_sentiment import analyze_stock_news
app = Flask(__name__)
app.secret_key = "supersecretkey"
DB = "sentiment.db"

LAST_ANALYZED_ARTICLES = []


@app.route("/", methods=["GET", "POST"])
def dashboard():
    global LAST_ANALYZED_ARTICLES
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    ticker_input = None
    article_limit = 100

    if request.method == "POST":
        ticker_input = request.form.get("ticker", "").upper()
        article_limit = request.form.get("limit", 100)

        try:
            article_limit = int(article_limit)
        except:
            article_limit = 100

        if ticker_input:
            LAST_ANALYZED_ARTICLES = analyze_stock_news(ticker_input, limit=article_limit)
            return redirect(url_for("dashboard"))

    analyzed_articles = LAST_ANALYZED_ARTICLES
    c.execute("""
        SELECT curr.ticker,
               curr.buy,
               curr.sell,
               curr.hold,
               curr.timestamp,
               (curr.buy - curr.sell) AS score,
               COALESCE((prev.buy - prev.sell), 0) AS prev_score
        FROM sentiment curr
        LEFT JOIN sentiment prev
            ON prev.id = (
                SELECT id FROM sentiment
                WHERE ticker = curr.ticker
                  AND id < curr.id
                ORDER BY id DESC
                LIMIT 1
            )
        INNER JOIN (
            SELECT ticker, MAX(id) AS max_id
            FROM sentiment
            GROUP BY ticker
        ) latest
        ON curr.id = latest.max_id
        ORDER BY curr.id DESC
    """)
    rows = c.fetchall()
    conn.close()

    html = """
    <style>
.spinner {
    display: none;
    border: 6px solid #f3f3f3;
    border-top: 6px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>

<div id="loadingSpinner" class="spinner"></div>
<h1>Stock Sentiment Dashboard</h1>
<form method="POST" style="margin-bottom:20px;" onsubmit="showSpinner()">
    <input type="text" name="ticker" placeholder="Enter Stock (e.g. TSLA)" required>
    <input type="number" name="limit" placeholder="News Limit" value="100" min="1">
    <button type="submit">Analyze</button>
</form>
<div style="display:flex; gap:20px; align-items:flex-start;">
    <div style="flex:1;">
    <table border=1 cellpadding=8 style="border-collapse: collapse;">
        <tr style="background-color:#222; color:white;">
            <th>Ticker</th>
            <th>BUY</th>
            <th>SELL</th>
            <th>HOLD</th>
            <th>Score</th>
            <th>Trend</th>
            <th>Signal</th>
            <th>Time</th>
        </tr>

        {% for r in rows %}
        <tr>
            <td><b>{{r[0]}}</b></td>
            <td style="color:green;">{{r[1]}}</td>
            <td style="color:red;">{{r[2]}}</td>
            <td>{{r[3]}}</td>
            <td><b>{{r[5]}}</b></td>
            <td>
                {% if r[5] > r[6] %}
                    <span style="color:green; font-weight:bold;">↑</span>
                {% elif r[5] < r[6] %}
                    <span style="color:red; font-weight:bold;">↓</span>
                {% else %}
                    <span style="color:gray;">→</span>
                {% endif %}
            </td>

            <td>
                {% if r[5] >= 5 %}
                    <span style="color:green; font-weight:bold;">
                        🚨 STRONG BUY
                    </span>
                {% elif r[5] <= -5 %}
                    <span style="color:red; font-weight:bold;">
                        🚨 STRONG SELL
                    </span>
                {% elif r[5] > 0 %}
                    <span style="color:green;">
                        Bullish
                    </span>
                {% elif r[5] < 0 %}
                    <span style="color:red;">
                        Bearish
                    </span>
                {% else %}
                    Neutral
                {% endif %}
            </td>

            <td>{{r[4]}}</td>
        </tr>
        {% endfor %}
    </table>
    </div>
    <div style="flex:1; max-height:700px; overflow-y:auto; border-left:1px solid #ccc; padding-left:15px;">
    <h3>Scanned Articles</h3>

    {% if analyzed_articles %}
        {% for item in analyzed_articles %}
            <div style="margin-bottom:15px; padding:10px; border-bottom:1px solid #ddd;">
                <b>Sentiment:</b>
                {% if item[1] == "BUY" %}
                    <span style="color:green; font-weight:bold;">{{item[1]}}</span>
                {% elif item[1] == "SELL" %}
                    <span style="color:red; font-weight:bold;">{{item[1]}}</span>
                {% else %}
                    {{item[1]}}
                {% endif %}
                <br><br>
                <div style="font-size:14px;">
                    {{item[0]}}
                </div>
            </div>
        {% endfor %}
    {% else %}
        <p>No articles analyzed yet.</p>
    {% endif %}
</div>
<script>
function showSpinner() {
    document.getElementById("loadingSpinner").style.display = "block";
    document.querySelector("button").innerText = "Analyzing...";
    document.querySelector("button").disabled = true;
}
</script>
</div>
    """
    return render_template_string(html, rows=rows, analyzed_articles=analyzed_articles)


@app.route("/latest/<ticker>")
def latest(ticker):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        SELECT buy, sell, hold, timestamp
        FROM sentiment
        WHERE ticker=?
        ORDER BY id DESC
        LIMIT 1
    """, (ticker,))
    row = c.fetchone()
    conn.close()

    if row:
        return jsonify({
            "ticker": ticker,
            "buy": row[0],
            "sell": row[1],
            "hold": row[2],
            "timestamp": row[3]
        })
    return jsonify({"error": "Not found"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4949)