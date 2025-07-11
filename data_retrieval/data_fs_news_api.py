import requests
import os
from datetime import timedelta
import json


def load_cached_data(save_dir: str, _def_fname: str) -> dict:
    try:
        with open(f'{save_dir}{_def_fname}', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Exception occured: {e}")


def fetch_data(**kwargs) -> dict:
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY environment variable not set")

    '''
    fetches data from the api
    source:newsapi.org
    returns: unprocessed json
    '''

    ticker = kwargs.get('ticker')
    timestamp = kwargs.get('timestamp')
    last_timestamp = timestamp - timedelta(days=14)

    save_dir = "./news_data/"
    _def_fname = "app_data.json"

    EVERYTHING_ENDPOINT_URI = f'https://newsapi.org/v2/everything?q={
        ticker}&from={last_timestamp}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'

    try:
        if (os.path.exists(f'{save_dir}{_def_fname}')):
            return load_cached_data(save_dir=save_dir, _def_fname=_def_fname)
        else:
            result = requests.get(EVERYTHING_ENDPOINT_URI)
            result.raise_for_status()
            save_json_to_file(result.json(), save_dir='./news_data/')
            return result.json()
    except Exception as e:
        print(f"Exception: {e}")


def save_json_to_file(data, filename="app_data.json", save_dir="./data/"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"JSON Data saved to {filename}")
    except Exception as e:
        print(f"Exception occured: {e}")


def return_titles(json_data):
    titles = []
    articles = json_data.get('articles', [])
    for article in articles:
        if article and 'title' in article:
            titles.append(article['title'])
    return titles
