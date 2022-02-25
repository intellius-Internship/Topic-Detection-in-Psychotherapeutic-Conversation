import os
import time

from numpy import sort
import textdistance
import pandas as pd

from glob import iglob
from tqdm import tqdm as tqdm_prog
from tqdm.auto import tqdm

from os.path import join as pjoin
from util import (parallelize_dataframe, load_reference)

from morph_analyzer import get_morphs
from approximate_nn import (build_model, load_index, load_transformer, get_nns)

'''
Description
-----------
데이터 증강에 필요한 파라미터 지정
    args.threshold: 벡터 간 거리 임계값 (값이 작을수록 유사함) 
    args.metric: metic of AnnoyIndexer
    args.cand_num: 최대 이웃 수 (후보군 최대 개수)
    args.use_tfidf: if True, use tfidf vector else use countvec
    args.model_dir: 모델 저장/로드 경로
    args.query: reference와 hypothesis 간 비교할 발화 
            - query: original utterance
            - query_pos: processed utterance (contains only nouns, adjectives, and verbs)
'''
def base_setting(args):
    args.threshold = getattr(args, 'threshold', 0.65)
    args.metric = getattr(args, 'metric', 'angular')
    args.cand_num = getattr(args, 'cand_num', 5)
    args.use_tfidf = getattr(args, 'use_tfidf', True)
    args.model_dir = getattr(args, 'model_dir', 'model')
    args.query = getattr(args, 'query', 'query_pos')

def get_counselling_data_by_annoy(kwargs, hypothesis):
    # load params
    args = kwargs['args']
    index_path = kwargs['index_path']
    reference = kwargs['reference']
    transformer = kwargs['transformer']
    cv = kwargs['cv']
    
    # load indexer
    index = load_index(args, index_path)

    # get approximate nearest neighbors
    tqdm.pandas(desc=f"get_nns (PID: {os.getpid()})", mininterval=0.01)
    hypothesis['candidates'] = hypothesis[args.query].progress_apply(lambda x: get_nns(cv=cv, index=index, \
        reference=reference, hypo_utter=x, n=args.cand_num, threshold=args.threshold, transformer=transformer))
    hypothesis['num_candidate'] = list(map(lambda x: len(x), hypothesis['candidates']))
    return hypothesis[hypothesis['num_candidate'] > 0]

'''
Description
-----------
발화의 후보 클래스들 중 가장 높은 스코어를 가지는 클래스로 라벨링
'''
def allocate_topic(data : pd.DataFrame):
    result = pd.DataFrame()
    for d in tqdm(data.iterrows(), total=len(data), desc='labeling topic'):
        row = d[1]
        candidate_turns = row.candidates.tolist()[0]
        candidate_turns = sorted(candidate_turns, key=lambda x: x[-1], reverse=True)

        dic = {
            'label_str': candidate_turns[0][0],
            'score': candidate_turns[0][-1],
            'query': row['query']
        }
        result = result.append(dic, ignore_index=True)
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

    result = pd.concat([result, etc_data], ignore_index=True)
    return result

def allocate_class_by_vector(args, hypothesis, reference): 
    base_setting(args)

    # analyze morphs
    if not ('query_pos' in hypothesis.columns and 'query_pos' in reference.columns):
        print(f"Split morphs")
        reference = get_morphs(reference, tokenizer=args.tokenizer)
        reference.to_csv(pjoin(args.data_dir, 'reference.csv'), index=False)

        hypothesis = get_morphs(hypothesis, tokenizer=args.tokenizer)
        hypothesis.to_csv(pjoin(args.data_dir, 'hypothesis.csv'), index=False)

    # build indexer
    if not list(iglob(pjoin(args.model_dir, 'indexer-*.annoy'), recursive=False)):
        print(f"Building Model")
        build_model(args=args, reference=reference[['query', 'query_pos', 'label_str']], use_tfidf=args.use_tfidf)

    # load model
    transformer, cv = load_transformer(args)
    reference = load_reference(args)
    indexer = list(iglob(pjoin(args.model_dir, 'indexer_*.annoy'), recursive=False))
    assert indexer
    index_path = indexer[0]

    # multiprocessing
    result = parallelize_dataframe(hypothesis, \
        get_counselling_data_by_annoy, num_cores=args.num_cores, args=args, \
            reference=reference, cv=cv, transformer=transformer, index_path=index_path)
    result = allocate_topic(result)

    # append etc class
    result = append_etc(args, result=result, hypothesis=hypothesis)

    # save result
    print(f"Number of Data: {len(result)}")
    result.to_csv(pjoin(args.result_dir, 'data_w_vector.csv'), index=False) 
    return