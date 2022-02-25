import os
import re
import multiprocessing
import pandas as pd
import numpy as np

from tqdm import tqdm
from os.path import join as pjoin
from util import parallelize_dataframe

'''
Description
-----------
주어진 리액션 클래스의 정규식 반환
'''
def get_regexp_list(topic_data, intent : str):
    return topic_data[topic_data.intent==intent]['regexp-list'].tolist()[0]

def get_regexp(topic_data, intent : str):
    regexp_list = get_regexp_list(topic_data=topic_data, intent=intent)
    return '|'.join(regexp_list)

'''
Description
-----------
주어진 리액션 클래스의 정규식에 포함되는 응답을 가진 대화 턴을 후보 턴으로 저장 
'''
def get_candidates(hypothesis : pd.DataFrame, topic_data : pd.DataFrame, intent : str, proc_query=True) -> list:
    regexp = get_regexp(topic_data=topic_data, intent=intent)
    entire_utter = hypothesis['proc_query'].tolist() + hypothesis['proc_reply'].tolist()
    entire_utter_original = hypothesis['query'].tolist() + hypothesis['reply'].tolist()
    idx = 0 if proc_query else 1
    try:
        candidate_utters = list(filter(lambda x: len(re.findall(regexp, x[idx])) > 0, zip(entire_utter, entire_utter_original)))
    except Exception as e:
        print(f"Error on topic: {intent} ({e})")

    return candidate_utters

def allocate_candidates(kwargs, topic_data : pd.DataFrame) -> pd.DataFrame:
    hypothesis = kwargs['hypothesis']
    reference = kwargs['reference']
    replies_by_topic = []

    # integrate reference and candidate dataset
    for intent in tqdm(topic_data.intent.unique(), total=len(topic_data.intent.unique()),
        desc=f"PID: {os.getpid()}", mininterval=0.01):
        candidate_utters = get_candidates(hypothesis=hypothesis, topic_data=topic_data, intent=intent)
        
        # '집중력저하' 클래스의 경우 hypothesis와 reference의 query 열 비교
        if intent == '집중력저하':
            candidate_utters += get_candidates(hypothesis=hypothesis, topic_data=topic_data, intent=intent, proc_query=False)

        candidate_utters = list(map(lambda x: x[-1], candidate_utters))
        candidate_utters += reference[reference.label_str == intent]['query'].tolist()
        candidate_utters = list(set(candidate_utters))
        replies_by_topic.append(candidate_utters)

    topic_data['candidate_utters'] = replies_by_topic
    topic_data['num_candidate'] = list(map(lambda x: len(x), topic_data['candidate_utters']))
    return topic_data

'''
Description
-----------
후처리 함수로 데이터 구조 변경
'''
def postprocess_candidates(topic_data : pd.DataFrame):
    entire_data = pd.DataFrame()

    labels = topic_data.intent.unique().tolist()
    for topic in tqdm(topic_data.intent.unique(), total=len(topic_data.intent.unique()), \
        desc=f"postprocess", mininterval=0.01):

        sub_data = pd.DataFrame()
        sub_data['query'] = topic_data[topic_data.intent==topic]['candidate_utters'].tolist()[0]
        sub_data['label'] = labels.index(topic)
        sub_data['label_str'] = topic

        entire_data = pd.concat([entire_data, sub_data], ignore_index=True)
        del sub_data

    return entire_data

'''
Description
-----------
기타 클래스 추가
'''
def append_etc(args, result : pd.DataFrame, hypothesis : pd.DataFrame):
    psychotherapic_utters = result['query'].tolist()
    etc_utters = []
    for utter in hypothesis['query']:
        if len(etc_utters) > args.num_etc:
            break
        if not utter in psychotherapic_utters:
            etc_utters.append(utter) 

    etc_data = pd.DataFrame()
    etc_data['query'] = etc_utters
    etc_data['label_str'] = '기타'
    etc_data['label'] = args.etc_label

    result = pd.concat([result, etc_data], ignore_index=True)
    return result


def allocate_class_by_regexp(args, hypothesis, reference, topic_data):
    # preprocessing query
    tqdm.pandas(desc="processing")
    hypothesis['proc_query'] = hypothesis['query'].progress_apply(lambda x: re.sub('[\s]+', ' ', re.sub('[^가-힣\s\?]+', ' ', x)))
    hypothesis['proc_reply'] = hypothesis['reply'].progress_apply(lambda x: re.sub('[\s]+', ' ', re.sub('[^가-힣\s\?]+', ' ', x)))

    # multiprocessing
    result = parallelize_dataframe(topic_data, allocate_candidates, \
        num_cores=args.num_cores, hypothesis=hypothesis, reference=reference)

    # post processing
    result = postprocess_candidates(result)

    # append etc class
    result = append_etc(args, result=result, hypothesis=hypothesis)

    # save result
    print(f"Number of Data: {len(result)}")
    result.to_csv(pjoin(args.result_dir, 'labeled_data_by_regexp.csv'), index=False)
    return 