import re
import argparse
import random
import pandas as pd
import numpy as np
import warnings
from functools import reduce
from os.path import join as pjoin
warnings.filterwarnings(action='ignore')

repeatchars_pattern = re.compile('(\D)\\1{2,}')
doublespace_pattern = re.compile('\s+')

def base_setting(args):
    args.shuffle = getattr(args, 'shuffle', False)
    args.seed = getattr(args, 'seed', 19)
    args.valid_ratio = getattr(args, 'valid_ratio', 0.2)

def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def del_newline(text : str):
    return re.sub('[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!~0-9a-zA-Z\s]+', '', text)

def preprocess(text : str):
    proc_txt = del_newline(text)
    proc_txt = del_special_char(proc_txt)
    proc_txt = repeat_normalize(proc_txt, num_repeats=3)

    return proc_txt.strip()

def is_valid(proc_text : str, threshold=2) -> bool:
    return len(re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ]', '', proc_text)) > threshold


def processing(args, data):
    base_setting(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'Original Length of Data : {len(data)}')

    data['proc_query'] = list(map(preprocess, data['query']))
    data.to_csv(pjoin(args.data_dir, 'data.csv'), index=False)
    return data

def split_dataset(args, data):
    if not args.shuffle:
        valid = pd.DataFrame()
        train = pd.DataFrame()
        test = pd.DataFrame()

        for idx in data.label.unique().tolist():
            sub_data = data[data.label==idx]
            num_valid = int(len(sub_data) * args.valid_ratio)

            if num_valid == 0:
                if len(sub_data) < 2:
                    train = pd.concat([train, sub_data], ignore_index=True)
                elif len(sub_data) < 3:
                    test = pd.concat([test, sub_data.iloc[:1]], ignore_index=True)
                    train = pd.concat([train, sub_data.iloc[1:]], ignore_index=True)
            else:
                valid = pd.concat([valid, sub_data.iloc[:num_valid]], ignore_index=True)
                test = pd.concat([test, sub_data.iloc[num_valid:2 * num_valid]], ignore_index=True)
                train = pd.concat([train, sub_data.iloc[2 * num_valid:]], ignore_index=True)

            del sub_data

        valid = valid.sample(frac=1, random_state=args.seed)
        test = test.sample(frac=1, random_state=args.seed)
        train = train.sample(frac=1, random_state=args.seed)
    else:
        data = data.sample(frac=1, random_state=args.seed)
        valid = data.iloc[:int(len(data) * args.valid_ratio)]
        test = data.iloc[int(len(data) * args.valid_ratio):2 * int(len(data) * args.valid_ratio)]
        train = data.iloc[2 * int(len(data) * args.valid_ratio):]

        print(f"Train Distribution : \n{train.label.value_counts()}")
        print(f"Valid Distribution : \n{valid.label.value_counts()}")
        print(f"Test Distribution : \n{test.label.value_counts()}")

    valid.to_csv(pjoin(args.data_dir, 'valid.csv'), index=False)
    test.to_csv(pjoin(args.data_dir, 'test.csv'), index=False)
    train.to_csv(pjoin(args.data_dir, 'train.csv'), index=False)

    print(f"Total Number of Data : {len(data)} -> {len(valid) + len(test) + len(train)}")
