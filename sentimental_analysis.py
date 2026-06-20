import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("social_media.csv")

# Sentiment Analysis Function
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Post"].apply(get_sentiment)

# Save results
df.to_csv("sentiment_output.csv", index=False)

# Sentiment counts
sentiment_counts = df["Sentiment"].value_counts()

positive = sentiment_counts.get("Positive", 0)
negative = sentiment_counts.get("Negative", 0)
neutral = sentiment_counts.get("Neutral", 0)

total = len(df)

# -------------------
# BAR CHART
# -------------------
plt.figure(figsize=(8,5))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()

# -------------------
# PIE CHART
# -------------------
plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%"
)
plt.title("Sentiment Percentage")
plt.savefig("pie_chart.png")
plt.close()

# -------------------
# WORD CLOUD
# -------------------
text = " ".join(df["Post"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("wordcloud.png")
plt.close()

# -------------------
# HTML REPORT
# -------------------
html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Sentiment Analysis Dashboard</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background:#f4f6f9;
    padding:20px;
}}

h1 {{
    text-align:center;
    color:#2c3e50;
}}

.card {{
    background:white;
    padding:20px;
    margin:20px auto;
    width:80%;
    border-radius:10px;
    box-shadow:0 0 10px rgba(0,0,0,0.1);
}}

.stats {{
    display:flex;
    justify-content:space-around;
    text-align:center;
}}

.stat {{
    padding:15px;
}}

img {{
    width:100%;
}}

</style>

</head>

<body>

<h1>Social Media Sentiment Analysis Dashboard</h1>

<div class="card">
<h2>Summary</h2>

<div class="stats">
<div class="stat">
<h3>{total}</h3>
<p>Total Posts</p>
</div>

<div class="stat">
<h3>{positive}</h3>
<p>Positive</p>
</div>

<div class="stat">
<h3>{negative}</h3>
<p>Negative</p>
</div>

<div class="stat">
<h3>{neutral}</h3>
<p>Neutral</p>
</div>
</div>

</div>

<div class="card">
<h2>Bar Chart</h2>
<img src="bar_chart.png">
</div>

<div class="card">
<h2>Pie Chart</h2>
<img src="pie_chart.png">
</div>

<div class="card">
<h2>Word Cloud</h2>
<img src="wordcloud.png">
</div>

<div class="card">
<h2>Conclusion</h2>
<p>
This report analyzes social media sentiments and categorizes posts into Positive,
Negative and Neutral groups using Natural Language Processing (NLP).
</p>
</div>

</body>
</html>
"""

with open("sentiment_report.html", "w", encoding="utf-8") as file:
    file.write(html)

print("Project completed successfully!")
print("Open sentiment_report.html from File Explorer.")