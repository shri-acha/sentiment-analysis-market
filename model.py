from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import matplotlib.pyplot as plt
import json
from collections import Counter, defaultdict
import torch
import yfinance
import pandas as pd
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")


def bpe_tokenizer(corpus, no_of_merges=10):
    tokens = set()
    merges = []
    vocab = get_vocab(corpus)
    for _ in range(no_of_merges):
        pairs = get_pair_freq(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        vocab = join_pair(best_pair, vocab)
        for word in vocab:
            tokens.update(word.split()) 
    print(tokens)
    return merges, vocab


def get_vocab(corpus):
    vocab = Counter()
    for word in corpus:
        # word += '</w>'
        chars = " ".join(list(word))
        vocab[chars] += 1
    return vocab


def get_pair_freq(vocab):
    pair_freq = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split(' ')
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pair_freq[(pair)] += 1
    return pair_freq


def join_pair(pair, vocab):
    pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
    replacement = ''.join(pair)
    new_vocab = {}
    for word in vocab:
        new_word = pattern.sub(replacement, word)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def load_corpus(file_path):
    corpus = []
    with open(file_path) as f:
        file_json = json.load(f)
        corpuses = [entry["text"].split(' ') for entry in file_json]
    for c_s in corpuses:
        for c__s in c_s:
            corpus.append(c__s)
    return corpus


model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone")


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["positive", "neutral", "negative"]
    return {label: float(prob) for label, prob in zip(labels, probs[0])}


def extract(df, label):
    return df["sentiment_scores"].apply(lambda x: x[label])
