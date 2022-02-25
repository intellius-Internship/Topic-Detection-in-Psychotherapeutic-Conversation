import os
import re
import pandas as pd

from tqdm.auto import tqdm as a_tqdm
from tqdm import tqdm

from functools import reduce
from util import parallelize_dataframe
from os.path import join as pjoin

'''
Description
-----------
한글과 물음표를 제외한 모든 문제 제거
'''
def get_core_speech(text):
    return re.sub('[^가-힣ㄱ-ㅎ?]', '', text)

def get_keywords(topic_data : pd.DataFrame, intent : str):
    topic = topic_data[topic_data['intent']==intent]
    assert len(topic) == 1
    return topic['kw-list'].tolist()[0]

'''
Description
-----------
키워드와 응답의 공백/특수문자/숫자/영문자를 삭제하여 키워드 매칭 여부를 판단하며
응답 내 키워드가 포함된 경우, 해당 대화 턴을 후보 턴으로 저장
'''
def get_candidates(hypothesis : pd.DataFrame, topic_data : pd.DataFrame, intent : str) -> list:
    keywords = get_keywords(topic_data=topic_data, intent=intent)
    proc_keywords = list(map(get_core_speech, keywords))
    
    candidate_utters = list(filter(lambda text: reduce(lambda x, y: x|y, list(map(lambda x: x in text[0], proc_keywords))), \
        zip(hypothesis['core_query'], hypothesis['query'])))
    candidate_utters = list(map(lambda x: x[-1], candidate_utters))
    return candidate_utters

def allocate_candidates(kwargs, topic_data : pd.DataFrame) -> pd.DataFrame:
    hypothesis = kwargs['hypothesis']
    reference = kwargs['reference']
    replies_by_topic = []

    # integrate reference and candidate dataset
    for intent in tqdm(topic_data.intent.unique(), total=len(topic_data.intent.unique()), \
        desc=f"PID: {os.getpid()}", mininterval=0.01):

        candidate_utters = get_candidates(hypothesis=hypothesis, topic_data=topic_data, intent=intent)
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
def postprocess_candidates(topic_data):
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

def allocate_class_by_keyword(args, hypothesis, reference, topic_data):
    # delete spaces, special characters, alphabets and numbers
    a_tqdm.pandas(desc='preprocessing utterance')
    hypothesis['core_query'] = hypothesis['query'].progress_apply(get_core_speech)

    # multiprocessing
    result = parallelize_dataframe(topic_data, allocate_candidates, \
        num_cores=args.num_cores, hypothesis=hypothesis, reference=reference)

    # post processing
    result = postprocess_candidates(result)

    # append etc class
    result = append_etc(args, result=result, hypothesis=hypothesis)

    # save result
    print(f"Number of Data: {len(result)}")
    result.to_csv(pjoin(args.result_dir, 'labeled_data_by_kw.csv'), index=False)
    return 