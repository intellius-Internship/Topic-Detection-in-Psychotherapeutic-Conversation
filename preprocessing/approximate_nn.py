import time
import pickle
import pandas as pd
from typing import List

from annoy import AnnoyIndex
from os.path import join as pjoin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

'''
Description
-----------
전체 발화 문장들을 count vector로 변환

Input
-----
    query_list: 발화 리스트
    cv: None 또는 CountVectorizer. 
        (None일 경우, CountVectorizer 생성)
Output
------
    vectors: query_list를 벡터로 변환한 값
    cv: CountVectorizer
'''
def get_countvec(query_list : List[str], cv=None):
    if cv is None:
        cv = CountVectorizer()
        vectors = cv.fit_transform(query_list)
        return cv, vectors
    else:
        vectors = cv.transform(query_list)
        return cv, vectors

'''
Description
-----------
전체 발화 문장들을 tfidf vector로 변환

Input
-----
    query_list: 발화 리스트
    cv: None 또는 CountVectorizer. 
    transformer: None 또는 TfidfTransformer. 
                (None일 경우, TfidfTransformer 생성)
Output
------
    vectors: query_list를 벡터로 변환한 값
    cv: CountVectorizer
    transformer: TfidfTransformer
'''
def get_tfidf(query_list : List[str], cv=None, transformer=None):
    if transformer is None:
        cv, vectors = get_countvec(query_list)
        transformer = TfidfTransformer()
        vectors = transformer.fit_transform(vectors)
        return transformer, vectors, cv
    else:
        assert cv is not None
        vectors = cv.transform(query_list)
        vectors = transformer.transform(vectors)
        return transformer, vectors, cv

'''
Description
-----------
AnnoyIndex 모델 빌드
'''
def build_model(args, reference : pd.DataFrame, use_tfidf=True):
    if not use_tfidf: # when args.use_tfidf is True
        cv, vectors = get_countvec(reference[args.query].tolist())
        print(f"Shape of countvec: {vectors.shape}")
    else:
        transformer, vectors, cv = get_tfidf(reference[args.query].tolist())
        print(f"Shape of tfidf: {vectors.shape}")

    vectors_len, vector_size = vectors.shape

    index = AnnoyIndex(vector_size, args.metric)
    for idx in range(vectors_len):
        vector = vectors[idx].toarray().squeeze(axis=0)
        index.add_item(idx, vector)
        del vector

    # build
    index.build(50)

    # save indexer
    index_path = pjoin(args.model_dir, f'indexer_{vector_size}.annoy')
    print(f"Saving {index_path}")
    index.save(index_path)
    
    print(f"Saving {pjoin(args.model_dir, 'reference.pickle')}")
    pickle.dump(reference.to_dict('records'), open(pjoin(args.data_dir, "reference.pickle"), "wb"))

    print(f"Saving {pjoin(args.model_dir, 'cv.pickle')}\n")
    pickle.dump(cv, open(pjoin(args.model_dir, "cv.pickle"), "wb"))
    
    if use_tfidf:
        pickle.dump(transformer, open(pjoin(args.model_dir, "transformer.pickle"), "wb"))
        print(f"Saving {pjoin(args.model_dir, 'transformer.pickle')}\n")

    return 

'''
Description
-----------
주어진 경로의 AnnoyIndex 로드
'''
def load_index(args, index_path):
    print(f"Load {index_path.split('/')[-1]}")
    vector_size = int(index_path.split('.')[0].split('_')[-1])
    index = AnnoyIndex(vector_size, args.metric)
    index.load(index_path)
    
    return index

'''
Description
-----------
주어진 경로의 CountVectorizer와 TfidfTransformer 로드
'''
def load_transformer(args):
    print(f"Load transformer.pickle")
    cv = pickle.load(open(pjoin(args.model_dir ,'cv.pickle'), "rb"))

    transformer = pickle.load(open(pjoin(args.model_dir ,'transformer.pickle'), "rb"))
    return transformer, cv

'''
Description
-----------
주어진 발화, hypo_utter에 대하여 대략적인 후보 클래스 추출
'''
def get_nns(reference : List[dict], hypo_utter : str, n : int, threshold : float, index, cv, transformer=None):
    # vectorization
    if transformer is None:
        _, vectors = get_countvec(query_list=[hypo_utter], cv=cv)
    else:
        _, vectors, _ = get_tfidf(query_list=[hypo_utter], cv=cv, transformer=transformer)
    vector = vectors[0].toarray().squeeze(axis=0)

    # get approximate nearest neighbors
    cand_ids, distances = index.get_nns_by_vector(vector, n, include_distances=True)
    candidates = list(map(lambda x: (reference[x[0]]['label_str'], reference[x[0]]['query'], x[-1]), \
        zip(cand_ids, distances)))

    # thresholding
    candidates = list(filter(lambda x: x[-1] < threshold, candidates))
    time.sleep(0.1)
    return candidates

