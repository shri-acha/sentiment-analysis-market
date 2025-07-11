import requests
import os
from datetime import timedelta
import json


def fetch_data(**kwargs):
    if (os.path.exists('./news_data/app_data.json')):
        try:
            with open('./news_data/app_data.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Exception occured: {e}")
    else:
        NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        ticker = kwargs.get('ticker')
        timestamp = kwargs.get('timestamp')
        last_timestamp = timestamp - timedelta(days=14)
        EVERYTHING_ENDPOINT_URI = f'https://newsapi.org/v2/everything?q={
            ticker}&from={last_timestamp}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'

        result = requests.get(EVERYTHING_ENDPOINT_URI)
        save_json_to_file(result.json(),save_dir='./news_data/')
        return result.json()


def save_json_to_file(data, filename="app_data.json",save_dir="./data/"):
    try:
        with open(save_dir + filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"JSON Data saved to {filename}")
    except Exception as e:
        print(f"Exception occured: {e}")

def relevant_data(json_data): # APPLICATION SPECIFIC CODE
    titles = []
    for article in json_data.get('articles'):
        titles.append(article['title'])
    return titles

