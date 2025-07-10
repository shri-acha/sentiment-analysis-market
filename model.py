import re
import json
from collections import Counter, defaultdict


class Tokenizer:
    def __init__(self, no_of_merges, corpus=[]):
    
        self.corpus = corpus
        self.tokens = set()
        self.no_of_merges = no_of_merges
        self.vocab = Counter()
        self.mapping_id = defaultdict(int)
        self.mapping_word = defaultdict(str)

        self.mapping_id["[CLS]"] = 304
        self.mapping_id["[SEP]"] = 305
        self.mapping_id["[PAD]"] = 306


        if not isinstance(corpus, (list, tuple)):
            raise TypeError("Corpus must be a list or tuple")
        if not isinstance(no_of_merges, int) or no_of_merges < 0:
            raise ValueError("Number of merges must be a non-negative integer")

    def bpe_tokenizer(self):
        merges = []
        self.update_vocab()
        for _ in range(self.no_of_merges):
            pairs = self.get_pair_freq()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)
            self.join_pair(best_pair)

        for word in self.vocab:
            self.tokens.update(word.split())

        # Sanity check for bad tokens
        for token in self.tokens:
            if token in {'</w', '</', '>'} or re.match(r'</?$', token):
                print("Bad token:", token)

        self.token_to_id()
        return merges, self.tokens

    def update_vocab(self):
        for word in self.corpus:
            word += '</w>'
            chars = " ".join(list(word))
            self.vocab[chars] += 1

    def get_pair_freq(self):
        pair_freq = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split(' ')
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freq[pair] += freq
        return pair_freq

    def join_pair(self, pair):
        pattern = re.compile(
            r'(?<!\S)' + re.escape(pair[0] + ' ' + pair[1]) + r'(?!\S)')
        replacement = pair[0] + pair[1]
        new_vocab = {}

        for word in self.vocab:
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = self.vocab[word]

        self.vocab = new_vocab

    def load_corpus(self, file_path):
        try:
            with open(file_path) as f:
                file_json = json.load(f)
                corpuses = [entry["text"].split(' ') for entry in file_json]
            for c_s in corpuses:
                for c__s in c_s:
                    self.corpus.append(c__s)
        except FileNotFoundError:
            raise FileNotFoundError(f"File Not Found!: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON Format: {e}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied trying to access: {file_path}")

    def token_to_id(self):
        for idx, token in enumerate(self.tokens):
            self.mapping_id[token] = idx
            self.mapping_word[idx] = token

    def encode(self, tokens_):
        return [self.mapping_id.get(token) for token in tokens_]

    def decode(self, ids):
        return [self.mapping_word.get(id_) for id_ in ids]



class _BERT:
    def __init__(self):
        pass
    def prepare_input(self):
        pass

