import requests
import json
import polars as pl
import csv
import os


from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")

API_KEY = "" #Enter API key here
url = (
     "https://newsapi.org/v2/top-headlines?"
)


params = {
    "category": "business",
    "country": "us",
    "pageSize": 100,
    "apiKey": API_KEY
}


response = requests.get(url, params=params)
data = response.json()

total_score = 0
num_articles = 0
articles_list = []

for article in data["articles"]:

    text = f"{article['title']} {article['description'] or ''}"
    sentiment = pipe(text)[0]


    print(f'Sentiment {sentiment["label"]}, Score: {sentiment["score"]}')
    
    if sentiment['label'] == 'positive': 
        total_score += sentiment['score']
        num_articles += 1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles +=1

    article_info = {
        "source_id": article["source"]["id"],
        "source_name": article["source"]["name"],
        "author": article["author"],
        "title": article["title"],
        "description": article["description"],
        "url": article["url"],
        "date_time": article["publishedAt"],
        "sentiment": sentiment["label"],
        "score": sentiment["score"]
    }

    articles_list.append(article_info)


for a in articles_list:
    print(a)

with open("articles.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["source_id", "source_name", "author", "title", "description", "url", "date_time", "sentiment", "score"])
    writer.writeheader()
    writer.writerows(articles_list)

    if num_articles > 0:
        avg_score = total_score / num_articles
        print(f"\nOverall sentiment: {avg_score:.4f}")

   


