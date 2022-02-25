import argparse
from os import mkdir
import warnings
import multiprocessing
import pandas as pd

from util import mkdir_p
from os.path import join as pjoin
from topic_allocator import allocate_class
from preprocess import processing, split_dataset


warnings.filterwarnings(action='ignore')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build Counselling Dataset')
    parser.add_argument('--split',
                        action='store_true',
                        default=False,
                        help='split dataset into train, valid, test')

    parser.add_argument('--preprocessing',
                        action='store_true',
                        default=False,
                        help='data preprocessing')

    parser.add_argument('--labeling',
                        type=str,
                        default= None,
                        choices=['keyword', 'textdist', 'regexp', 'vector'],
                        help='method of reation labeling')

    parser.add_argument('--data_dir',
                        type=str,
                        default='../data')

    parser.add_argument('--result_dir',
                        type=str,
                        default='../result')

    parser.add_argument('--num_cores',
                        type=int,
                        default=multiprocessing.cpu_count())

    args = parser.parse_args()

    mkdir_p(args.result_dir)
    if args.labeling is not None:
        allocate_class(args)

    try:
        data = pd.read_csv(pjoin(args.data_dir, 'data.csv')).dropna(axis=0)
        if args.preprocessing:
            data = processing(args, data)
        
        if args.split:
            split_dataset(args, data)
    except FileNotFoundError as e:
        print(e)
        pass