import warnings
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.model_util import U_TKN, S_TKN

warnings.filterwarnings(action='ignore')

DELIMITER = '<unused1>'

class PlmData(Dataset):
    """Dataloader for Topic-Detection Model based on Transformer"""
    def __init__(self, data_path, tokenizer, max_len):
        self._data = pd.read_csv(data_path, encoding='utf-8')
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(self.tokenizer.cls_token + \
            str(text) + self.tokenizer.sep_token)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids, len(ids)

    def _padding(self, ids):
        # padding with 'pad_token_id' of tokenizer
        while len(ids) < self.max_len:
            ids += [self.tokenizer.pad_token_id]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len-1] + [ids[-1]]
        return ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        
        query = turn['proc_query'] 
        label = int(turn['label'])

        token_ids, _ = self._tokenize(query)
        token_ids = self._padding(token_ids)

        attention_masks = [float(id>0) for id in token_ids]
        return(token_ids, np.array(attention_masks), label)


class AutoRegressionChatData(Dataset):
    """Dataloader for Topic-Detection Model based on GPT2"""
    def __init__(self, data_path, tokenizer, max_len):
        self._data = pd.read_csv(data_path, sep=',')
        self._data = self._data.dropna(axis=0)
        
        self.usr_token = U_TKN
        self.sys_token = S_TKN

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens, len(tokens)

    def _tokenize_turn(self, query, relation):
        query_toked, query_len = self._tokenize(self.usr_token + str(query))
        relation_toked, relation_len = self._tokenize(self.sys_token + str(relation) + self.tokenizer.eos_token)
        
        if query_len + relation_len > self.max_len:
            remain = self.max_len - query_len
            if remain <= 0:
                # Query가 max_len을 넘어가는 경우, max_len의 반절로 제한
                query_toked = [query_toked[0]] + query_toked[-(int(self.max_len/2))+1:] 
                query_len = len(query_toked)
                remain = self.max_len - query_len
                assert remain > 0

            relation_toked = relation_toked[:remain-1]+[relation_toked[-1]]
            relation_len = len(relation_toked)

        return query_toked, relation_toked, query_len, relation_len
        
    def _padding(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding with 'pad_token_id' of tokenizer
        while len(ids) < self.max_len:
            ids += [self.tokenizer.pad_token_id]
        return ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        
        query = turn['proc_query']
        label = turn['label_str']

        query_toked, label_toked, query_len, label_len = self._tokenize_turn(query, label)
        
        labels = [
            self.tokenizer.mask_token,
        ] * query_len + label_toked[1:]

        labels_ids = self._padding(labels)
        token_ids = self._padding(query_toked + label_toked)
        mask = [0] * query_len + [1] * label_len + [0] * (self.max_len - query_len - label_len)

        return(token_ids, np.array(mask), labels_ids)

class Seq2SeqChatData(Dataset):
    """Dataloader for Topic-Detection Model based on BART"""
    def __init__(self, data_path, tokenizer, max_len) -> None:
        self._data = pd.read_csv(data_path, sep=',')
        self._data = self._data.dropna(axis=0)

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)

        # padding with zeros
        if len(input_id) < self.max_len:
            while len(input_id) < self.max_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        turn = self._data.iloc[index]
        
        query = turn['proc_query']
        label = turn['label_str']
        
        query_toked = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(query) + [self.tokenizer.eos_token]
        label_toked = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(label) + [self.tokenizer.eos_token]

        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            query_toked, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            label_toked, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            label_toked[1:(self.max_len + 1)])

        # padding with negative values
        if len(labels) < self.max_len:
            while len(labels) < self.max_len:
                labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}



