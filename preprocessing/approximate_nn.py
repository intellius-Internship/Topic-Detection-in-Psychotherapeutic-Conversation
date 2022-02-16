import pickle
import pandas as pd
from typing import List
from glob import iglob

from tqdm import tqdm
from annoy import AnnoyIndex
from os.path import join as pjoin
from sklearn.feature_extraction.text import CountVectorizer

def get_countvec(query_list : List[str], cv=None):
    if cv is None:
        cv = CountVectorizer()
        vectors = cv.fit_transform(query_list)
        return cv, vectors
    else:
        vectors = cv.transform(query_list)
        return cv, vectors

def build_model(args, ref_data : pd.DataFrame):
    cv, vectors = get_countvec(ref_data[args.query].tolist())
    # print(f"Shape of countvec: {vectors.shape}")

    vectors_len, vector_size = vectors.shape

    index = AnnoyIndex(vector_size, args.metric)
    for idx in range(vectors_len):
        vector = vectors[idx].toarray().squeeze(axis=0)
        index.add_item(idx, vector)
        del vector

    index.build(50)
    index_path = pjoin(args.model_dir, f'indexer_{vector_size}.annoy')
    index.save(index_path)
    pickle.dump(ref_data.to_dict('records'), open(pjoin(args.data_dir, "reference.pickle"), "wb"))
    pickle.dump(cv, open(pjoin(args.model_dir, "cv.pickle"), "wb"))

    print(f"Saving {index_path}")
    print(f"Saving {pjoin(args.model_dir, 'reference.pickle')}")
    print(f"Saving {pjoin(args.model_dir, 'cv.pickle')}\n")

    return 

def load_model(args):
    indexer = list(iglob(pjoin(args.model_dir, 'indexer_*.annoy'), recursive=False))
    assert indexer

    index_path = indexer[0]
    print(f"Load {index_path.split('/')[-1]}")

    vector_size = int(index_path.split('.')[0].split('_')[-1])
    index = AnnoyIndex(vector_size, args.metric)
    index.load(index_path)

    cv = pickle.load(open(pjoin(args.model_dir ,'cv.pickle'), "rb"))
    return index, cv

def get_nns(cv, index, ref_data : List[dict], hypothesis : str, n : int, threshold : float):
    _, vectors = get_countvec(query_list=[hypothesis], cv=cv)
    vector = vectors[0].toarray().squeeze(axis=0)
    cand_ids, distances = index.get_nns_by_vector(vector, n, include_distances=True)
    candidates = list(map(lambda x: (ref_data[x[0]]['label_str'], ref_data[x[0]]['query'], x[-1]), \
        zip(cand_ids, distances)))
    candidates = list(filter(lambda x: x[-1] < threshold, candidates))
    return candidates

