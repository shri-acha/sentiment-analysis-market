from model import get_sentiment, extract, bpe_tokenizer, load_corpus
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    data = [
        {"timestamp": "2025-06-12 09:00",
            "text": "Fed signals pause on interest rate hikes amid inflation slowdown"},
        {"timestamp": "2025-06-12 11:00", "text": "Markets rally as tech stocks soar"},
        {"timestamp": "2025-06-12 14:00",
            "text": "Oil prices drop after OPEC decision"},
        {"timestamp": "2025-06-13 09:00",
            "text": "Weak job report sparks recession fears"},
        {"timestamp": "2025-06-13 12:00",
            "text": "Central banks warn of persistent economic risks"}
    ]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Hello from sentiment-analysis-market!")

    df["sentiment_scores"] = df["text"].apply(get_sentiment)
    df["positive"] = extract(df, "positive")
    df["neutral"] = extract(df, "neutral")
    df["negative"] = extract(df, "negative")

    df["date"] = df["timestamp"].dt.date
    daily_sentiment = df.groupby(
        "date")[["positive", "neutral", "negative"]].mean()
    print(daily_sentiment)

    corpus = load_corpus('./corpus/financial_corpus.json')
    # corpus = ["low", "lower", "lowest","lowering"]
    # print(corpuses)
    merges,vocab = bpe_tokenizer(corpus, 50)


if __name__ == '__main__':
    main()
