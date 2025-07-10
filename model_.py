from transformers import BertTokenizer, BertForSequenceClassification
import kagglehub
import pandas as pd
import torch
import metric
import AutoTrainer


class _finBERT:
    def __init__(self):

        metric = evaluate.load("accuracy")

        self.tokenizer = BertTokenizer.from_pretrained(
            'yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased")
        self.path = kagglehub.dataset_download(
            "aravsood7/sentiment-analysis-labelled-financial-news-data")

    def prepare_input(self):
        pass

    def tokenize_function(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True)

    def trainer(self):
        pass


    def compute_metrics(eval_pred):
       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       return self.metric.compute(predictions=predictions, references=labels)

    def test(self):
        training_data = pd.read_csv(self.path+'/Fin_Cleaned.csv')
        _actual_training_data = pd.DataFrame({
            'f_text': training_data['Headline'], 'sentiment': training_data['Final Status']
        })
        tokenized_data = self.tokenizer(
            _actual_training_data['f_text'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tokenized_ds = _actual_training_data.map(self.tokenizer).shuffle(seed=69)
 
        output = self.model(**tokenized_data)

        predicted_classes = torch.argmax(output.logits, dim=1)
        print(predicted_classes)
