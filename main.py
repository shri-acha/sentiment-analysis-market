from model import Tokenizer
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_ import _finBERT 
from dotenv import load_dotenv


def main():
    # data = [
    #     {"timestamp": "2025-06-12 09:00",
    #         "text": "Fed signals pause on interest rate hikes amid inflation slowdown"},
    #     {"timestamp": "2025-06-12 11:00", "text": "Markets rally as tech stocks soar"},
    #     {"timestamp": "2025-06-12 14:00",
    #         "text": "Oil prices drop after OPEC decision"},
    #     {"timestamp": "2025-06-13 09:00",
    #         "text": "Weak job report sparks recession fears"},
    #     {"timestamp": "2025-06-13 12:00",
    #         "text": "Central banks warn of persistent economic risks"}
    # ]
    #
    # df = pd.DataFrame(data)
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    #
    # print("Hello from sentiment-analysis-market!")

    # df["sentiment_scores"] = df["text"].apply(get_sentiment)
    # df["positive"] = extract(df, "positive")
    # df["neutral"] = extract(df, "neutral")
    # df["negative"] = extract(df, "negative")

    # df["date"] = df["timestamp"].dt.date
    # daily_sentiment = df.groupby(
    #     "date")[["positive", "neutral", "negative"]].mean()
    # print(daily_sentiment)

    # corpus = ["This", "is", "a", "transformer"]
    # tokenizer = Tokenizer(100)
    # tokenizer.load_corpus('./corpus/financial_corpus.json')
    # # print(corpuses)
    # merges, vocab = tokenizer.bpe_tokenizer()
    # f = os.open("./vocab.txt", 777)

    load_dotenv()
    model = _finBERT() 
    # Run analysis
    # results = model.run(company='google', show_details=True)
    
    # You can also analyze multiple companies
    companies = ['google', 'apple', 'microsoft']
    for company in companies:
        company_results = model.run(company=company, show_details=False)
        print(f"{company_results['overall_sentiment']}")



if __name__ == '__main__':
    main()
