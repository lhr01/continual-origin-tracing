import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class COT_Bench():
    use_path = True

    class_order = np.arange(20).tolist()

    def download_data(self, dataset_name):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/{}/train.jsonl".format(dataset_name)
        dev_dir = "./data/{}/dev.jsonl".format(dataset_name)
        test_dir = "./data/{}/test.jsonl".format(dataset_name)

        train_dset = pd.read_json(train_dir, lines=True)
        dev_dset = pd.read_json(dev_dir, lines=True)
        test_dset = pd.read_json(test_dir, lines=True)

        self.train_data = train_dset['text'].values
        self.train_targets = train_dset['label'].values

        self.dev_data = dev_dset['text'].values
        self.dev_targets = dev_dset['label'].values

        self.test_data = test_dset['text'].values
        self.test_targets = test_dset['label'].values

        self.class_to_idx = train_dset.set_index('model')['label'].to_dict()
        self.classes = train_dset['model'].unique()


class textsDataset(Dataset):

    def __init__(self, tokenizer, texts, labels, max_seq_length = 512):
        super(textsDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(texts, labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]
    
    def get_input(self, texts, labels):
        # tokenizer  (list of shape [text_len, token_len])
        tokens_text = list(map(self.tokenizer.tokenize, texts))

        result = list(map(self.truncate_and_pad, tokens_text))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        tokens_type_ids = [i[2] for i in result]

        return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(tokens_type_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )

    def truncate_and_pad(self, tokens_text):
        
        tokens_text = ['[cls]'] + tokens_text

        if len(tokens_text) > self.max_seq_length:
            tokens_text = tokens_text[0 : self.max_seq_length]

        padding = [0] * (self.max_seq_length - len(tokens_text))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_text)
        input_ids += padding

        attention_mask = [1] * len(tokens_text) + padding

        token_type_ids = [0] * (self.max_seq_length)

        assert len(input_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(token_type_ids) == self.max_seq_length

        return input_ids, attention_mask, token_type_ids


class LlamaDataset(Dataset):

    def __init__(self, tokenizer, texts, labels, max_seq_length=512):
        super(LlamaDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.tokenizer.pad_token = tokenizer.eos_token
        self.input_ids, self.attention_mask, self.labels = self.get_input(texts, labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]
    
    def get_input(self, texts, labels):
        """
        Tokenize texts and convert them to input IDs and attention masks for LLaMA.
        """
        texts = texts.tolist()
        encodings = self.tokenizer(
            texts,
            padding="max_length",  # 自动填充到 max_seq_length
            truncation=True,       # 截断超出 max_seq_length 的部分
            max_length=self.max_seq_length,
            return_tensors="pt"    # 返回 PyTorch 张量
        )

        # 返回 input_ids, attention_mask 和 labels
        return (
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long)
        )