import pandas as pd

from ast import literal_eval
from os.path import join as pjoin

from allocator_by_regexp import allocate_class_by_regexp
from allocator_by_textdistance import allocate_class_by_textdistance
from allocator_by_keyword import allocate_class_by_keyword
from allocator_by_vector import allocate_class_by_vector

'''
Description
-----------
파라미터 추가
    args.num_etc: '기타' 클래스 데이터 개수
'''
def base_setting(args):
    args.tokenizer = getattr(args, 'tokenizer', 'mecab')
    args.num_etc = getattr(args, 'num_etc', 10001)
    args.etc_label = getattr(args, 'etc_label', 19)

'''
Description
-----------
hypothesis 중에서 reference와 비슷한 발화 수집함으로써 \
     학습 데이터 증강
     
topic labeling 함수
    def allocate_class_by_textdistance \
        -> 텍스트 유사도 기반 상담주제 라벨링
    def allocate_class_by_regexp \
        -> 정규식 매칭 기반 상담주제 라벨링
    def allocate_class_by_keyword \
        -> 키워드 기반 상담주제 라벨링
    def allocate_class_by_vector \
        -> 벡터 기반 상담주제 라벨링

    reference data: 심리상담 대화 데이터
    hypothesis data: 일상 대화 데이터
'''
def allocate_class(args):
    base_setting(args)

    # raw dialogue dataset
    hypothesis = pd.read_csv(pjoin(args.data_dir, 'hypothesis.csv'))
    hypothesis.drop_duplicates(['query'], inplace=True)
    
    # topic-dialogue dataset
    reference = pd.read_csv(pjoin(args.data_dir, 'reference.csv'))
    reference.drop_duplicates(['query'], inplace=True)

    # topic-kw-regexp dataset
    topic_data = pd.read_csv(pjoin(args.data_dir, 'topic.csv'), converters={
        "kw-list": literal_eval,
        "regexp-list" : literal_eval
    })

    print(f"Number of Cores: {args.num_cores}")
    if args.labeling == 'textdist':
        allocate_class_by_textdistance(args=args, hypothesis=hypothesis, reference=reference)
    elif args.labeling == 'regexp':
        allocate_class_by_regexp(args=args, hypothesis=hypothesis, reference=reference, topic_data=topic_data)
    elif args.labeling == 'keyword':
        allocate_class_by_keyword(args=args, hypothesis=hypothesis, reference=reference, topic_data=topic_data)
    elif args.labeling == 'vector':
        allocate_class_by_vector(args=args, hypothesis=hypothesis, reference=reference)

    return