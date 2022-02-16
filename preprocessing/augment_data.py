import textdistance

import pandas as pd

from glob import iglob
from tqdm import tqdm

from os.path import join as pjoin
from util import (parallelize_dataframe, \
    load_hypothesis, load_reference)

from morph_analyzer import get_morphs
from approximate_nn import (build_model, load_model, get_nns)

def base_setting(args):
    args.threshold = getattr(args, 'threshold', 0.65)
    args.build = getattr(args, 'build', False)
    args.metric = getattr(args, 'metric', 'angular')
    args.cand_num = getattr(args, 'cand_num', 5)
    args.model_dir = getattr(args, 'model_dir', 'model')
    args.query = getattr(args, 'query', 'query_pos')


def get_candidate_turns_by_textdistance(data, query, threshold):
    similar_turn = data['query'].apply(lambda x: (1-textdistance.jaro_winkler.normalized_distance(x, query), x))
    similar_turn = sorted(similar_turn, key=lambda x: x[0], reverse=True)
    similar_turn = list(filter(lambda x: x[0] > threshold, similar_turn))
    return similar_turn

def get_counselling_data_by_textdistance(kwargs, ref_data):
    hypo_data = kwargs['hypo_data']
    args = kwargs['args']
    
    entire_data = pd.DataFrame()
    for d in tqdm(ref_data.iterrows(), total = len(ref_data)):
        row = d[1]
        candidates = get_candidate_turns_by_textdistance(data=hypo_data, \
            query=row['query'], threshold=args.threshold)
        
        cand_data = pd.DataFrame()
        cand_data['score'] = list(map(lambda x: x[0], candidates))
        cand_data['query'] = list(map(lambda x: x[1], candidates))
        cand_data['label_str'] = row.label_str
        cand_data['label'] = row.label
        entire_data = pd.concat([entire_data, cand_data], ignore_index=True)
    
    return entire_data

def get_counselling_data_by_annoy(kwargs, hypo_data):
    args = kwargs['args']
    
    index, cv = load_model(args)
    ref_data = load_reference(args)
    
    hypo_data['candidate'] = hypo_data[args.query].apply(lambda x: get_nns(cv, index, \
        ref_data=ref_data, hypothesis=x, n=args.cand_num, threshold=args.threshold))
    return hypo_data

def augmentation(args): 
    base_setting(args)

    hypo_data = pd.read_csv(pjoin(args.data_dir, 'hypothesis.csv'))
    ref_data = pd.read_csv(pjoin(args.data_dir, 'data.csv'))

    hypo_data.drop_duplicates(['query'], inplace=True)
    ref_data.drop_duplicates(['query'], inplace=True)

    print(f"Number of Cores: {args.num_cores}")

    if args.use_textdistance:
        print(f"Augmentation with textdistance")
        
        result = parallelize_dataframe(ref_data, \
            get_counselling_data_by_textdistance, num_cores=args.num_cores, args=args, hypo_data=hypo_data)
        result.to_csv(pjoin(args.result_dir, 'data_w_textdistance.csv'), index=False) 

    else:
        if not ('query_pos' in hypo_data.columns and 'query_pos' in hypo_data.columns):
            print(f"Split morphs")
            ref_data = get_morphs(ref_data)
            ref_data.to_csv(pjoin(args.data_dir, 'data.csv'), index=False)

            hypo_data = get_morphs(hypo_data)
            hypo_data.to_csv(pjoin(args.data_dir, 'hypothesis.csv'), index=False)

        if not list(iglob(pjoin(args.model_dir, 'indexer-*.annoy'), recursive=False)):
            print(f"Building Model")
            build_model(args=args, ref_data=ref_data[['query', 'query_pos', 'label_str']])

        result = parallelize_dataframe(hypo_data, \
            get_counselling_data_by_annoy, num_cores=args.num_cores, args=args)
        result['cand_num'] = list(map(lambda x: len(x), result['candidate']))
        result = result[result['cand_num'] > 0]
        result.to_csv(pjoin(args.result_dir, 'data_w_annoy.csv'), index=False) 

        print(f"Number of Data: {len(result)}")
    
    return result