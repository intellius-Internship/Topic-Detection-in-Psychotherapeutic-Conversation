import os
import errno

from shutil import rmtree

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def del_folder(path):
    try:
        rmtree(path)
    except:
        pass

def tokenize(sent, tokenizer, max_len):
    tokens = tokenizer.tokenize(sent)
    seq_len = len(tokens)
    if seq_len > max_len:
        tokens = tokens[:max_len-1] + [tokens[-1]]
        seq_len = len(tokens)
        
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(token_ids)

    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]
        attention_mask += [0]

    return token_ids, attention_mask


def encode(sent, tokenizer, max_len):
    tok_ids, attention_mask = tokenize(sent, tokenizer=tokenizer, max_len=max_len)
    return tok_ids, attention_mask