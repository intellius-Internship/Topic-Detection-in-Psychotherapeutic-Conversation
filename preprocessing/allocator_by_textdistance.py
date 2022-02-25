import os
import time
import textdistance
import pandas as pd

from tqdm import tqdm as tqdm_prog
from tqdm.auto import tqdm

from os.path import join as pjoin
from typing import List
from util import parallelize_dataframe

'''
Description
-----------
텍스트 유사도 기반 라벨링에 필요한 파라미터 지정

    args.threshold: 텍스트 유사도 기반 유사도 점수 임계값 
    args.algorithm: 텍스트 유사도 알고리즘
'''
def base_setting(args):
    args.threshold = getattr(args, 'threshold', 0.8)
    args.algorithm = getattr(args, 'algorithm', 'jaro_winkler')

'''
Description
-----------
텍스트 유사도 기반 유사도 점수 계산 (0 to 1)
'''
def get_similar_score(hypo, ref, algorithm):
    if algorithm == 'jaro_winkler':
        return 1-textdistance.jaro_winkler.normalized_distance(hypo, ref)
    if algorithm == 'levenshtein':
        return 1-textdistance.levenshtein.normalized_distance(hypo, ref)
    if algorithm == 'hamming':
        return 1-textdistance.hamming.normalized_distance(hypo, ref)
    
    raise NotImplementedError('Not Implemented')

'''
Description
-----------
상담 발화인 u_utter과 유사한 응답을 가지는 발화 추출
'''
def get_candidate_turns(data, u_utter, threshold : float, algorithm : str):
    tqdm.pandas(desc=f"get_candidate_turns (PID: {os.getpid()})", mininterval=0.01)
    similar_turn = data['query'].progress_apply(lambda x: (get_similar_score(x, u_utter, algorithm), x))
    similar_turn = sorted(similar_turn, key=lambda x: x[0], reverse=True)
    similar_turn = list(filter(lambda x: x[0] > threshold, similar_turn))
    return similar_turn

'''
Description
-----------
topic 클래스의 기존 상담 발화 데이터인 utters와 \
    유사한 발화 후보 추출 및 dataframe 생성
'''
def allocate_candidates(kwargs, reference):
    hypothesis = kwargs['hypothesis']
    args = kwargs['args']
    
    entire_data = pd.DataFrame()
    for d in tqdm_prog(reference.iterrows(), total = len(reference), desc=f"PID: {os.getpid()}", mininterval=0.01):
        row = d[1]
        candidates = get_candidate_turns(data=hypothesis, \
            u_utter=row['query'], threshold=args.threshold, algorithm=args.algorithm)
        
        cand_data = pd.DataFrame()
        cand_data['score'] = list(map(lambda x: x[0], candidates))
        cand_data['query'] = list(map(lambda x: x[1], candidates))
        cand_data['label_str'] = row.label_str
        cand_data['label'] = row.label

        entire_data = pd.concat([entire_data, cand_data], ignore_index=True)
        del cand_data, candidates
        time.sleep(0.1)
    
    return entire_data

'''
Description
-----------
각 대화 턴의 후보 클래스 중 가장 높은 스코어를 가지는 클래스로 라벨링
'''
def allocate_topic(data : pd.DataFrame):
    result = pd.DataFrame()
    data_wo_duplicates = data.drop_duplicates(['query'])

    for d in tqdm(data_wo_duplicates.iterrows(), total=len(data_wo_duplicates), desc='labeling topic'):
        row = d[1]
        sub_data = data[data['query'] == row['query']]
        sub_data.sort_values(by=['score'], axis=0, inplace=True, ascending=False)
        result = pd.concat([result, sub_data.iloc[:1]], ignore_index=True)
        
    return result

'''
Description
-----------
기타 클래스 추가
'''
def append_etc(args, result : pd.DataFrame, hypothesis : pd.DataFrame):
    psychotherapic_utters = result['query'].tolist()
    etc_utters = list(filter(lambda x: not x in psychotherapic_utters, \
        hypothesis['query'].tolist()))
    etc_utters = etc_utters[:args.num_etc]   

    etc_data = pd.DataFrame()
    etc_data['query'] = etc_utters
    etc_data['score'] = 0.0
    etc_data['label_str'] = '기타'
    etc_data['label'] = args.etc_label

    result = pd.concat([result, etc_data], ignore_index=True)
    return result

def allocate_class_by_textdistance(args, hypothesis, reference):
    base_setting(args)

    # multiprocessing
    result = parallelize_dataframe(reference, \
        allocate_candidates, num_cores=args.num_cores, args=args, hypothesis=hypothesis)
    result = allocate_topic(result)

    # append etc class
    result = append_etc(args, result=result, hypothesis=hypothesis)

    # save result
    print(f"Number of Data: {len(result)}")
    result.to_csv(pjoin(args.result_dir, 'labeled_data_by_textdist.csv'))
    return