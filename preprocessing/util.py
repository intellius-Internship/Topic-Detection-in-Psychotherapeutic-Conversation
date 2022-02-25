import os
import errno
import pickle
import numpy as np
import pandas as pd

from os.path import join as pjoin
from shutil import rmtree
from multiprocessing import Pool
from functools import partial

'''
Description
-----------
멀티 프로세싱을 수행하기 위한 함수로, 
data를 num_cores 개로 분할하여 각 프로세스 별로 func 수행
'''
def parallelize_dataframe(data, func, num_cores, **kwargs):
    sub_data = np.array_split(data, num_cores)
    pool = Pool(num_cores)

    data = pd.concat(pool.map(partial(func, kwargs), sub_data))
    pool.close()
    pool.join()
    return data

'''
Description
-----------
<data_dir>/reference.pickle 로드
'''
def load_reference(args):
    print(f'Load Reference')
    return pickle.load(open(pjoin(args.data_dir ,'reference.pickle'), "rb"))

'''
Description
-----------
디렉토리 삭제 및 생성
'''
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