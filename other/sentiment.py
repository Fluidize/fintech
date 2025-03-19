from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import requests
import praw

#webscraper
reddit_read_only = praw.Reddit(client_id="bluyurav2bS2T4r46r-Abg", client_secret="E5M-dK9kA9TGlWqynZ2j6rXPKx-5oA", user_agent="Careless-Bus-5168")
subreddit = reddit_read_only.subreddit("rugpull")

data = []
for post in subreddit.hot(limit=100):
    data.append(post.title)

results = []
sia = SentimentIntensityAnalyzer()
for sentence in data:
    score = sia.polarity_scores(sentence)
    results.append(score['compound'])

print(results)

avg = 0
count = 0
for result in results:
    if result != 0.0:
        avg += result
        count += 1

avg = avg/count
print(avg)