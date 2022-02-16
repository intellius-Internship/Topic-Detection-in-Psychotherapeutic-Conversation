import argparse
import warnings
import multiprocessing
import pandas as pd

from os.path import join as pjoin
from augment_data import augmentation
from preprocess import processing, split_dataset


warnings.filterwarnings(action='ignore')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build Counselling Dataset')
    parser.add_argument('--augmentation',
                        action='store_true',
                        default=False)

    parser.add_argument('--use_textdistance',
                        action='store_true',
                        default=False)

    parser.add_argument('--split',
                        action='store_true',
                        default=False)

    parser.add_argument('--data_dir',
                        type=str,
                        default='data')

    parser.add_argument('--result_dir',
                        type=str,
                        default='result')

    parser.add_argument('--num_cores',
                        type=int,
                        default=multiprocessing.cpu_count())

    args = parser.parse_args()
    
    data = pd.read_csv(pjoin(args.data_dir, 'counselling_data.csv')).dropna(axis=0)
    data.rename( {'utter':'query', 'intent':'label_str'}, axis='columns', inplace=True)

    data = processing(args, data)

    if args.split:
        split_dataset(args, data)

    if args.augmentation:
        data = augmentation(args)

    
    