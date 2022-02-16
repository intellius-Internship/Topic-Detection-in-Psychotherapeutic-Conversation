import os
import errno
import pickle
import numpy as np
import pandas as pd

from os.path import join as pjoin
from shutil import rmtree
from multiprocessing import Pool
from functools import partial

def parallelize_dataframe(data, func, num_cores, **kwargs):
    sub_data = np.array_split(data, num_cores)
    pool = Pool(num_cores)

    data = pd.concat(pool.map(partial(func, kwargs), sub_data))
    pool.close()
    pool.join()
    return data

def load_reference(args):
    return pickle.load(open(pjoin(args.data_dir ,'reference.pickle'), "rb"))

def load_hypothesis(args):
    # return pickle.load(open(pjoin(args.data_dir ,'hypothesis.pickle'), "rb"))
    return pd.read_csv(pjoin(args.data_dir, 'hypothesis.csv'))

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