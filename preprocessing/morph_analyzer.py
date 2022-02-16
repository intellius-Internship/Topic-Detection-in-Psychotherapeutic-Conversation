import pandas as pd

from typing import List
from konlpy.tag import Okt 
from konlpy.tag import Mecab

mecab = Mecab()
okt = Okt()

TOKENIZER = 'mecab'
SEED = 19

def concat_pos(pos_list : List[tuple], concat_pos_list : List[tuple]) -> List[tuple]:
    if not pos_list:
        return concat_pos_list

    if len(concat_pos_list)>0 and concat_pos_list[-1][1].startswith(('XSV', 'VV', 'VA')) and pos_list[0][1] in ['EC', 'EP', 'EF', 'ETM']:
        concat_pos_list[-1] = (concat_pos_list[-1][0]+pos_list[0][0],
                            f"{concat_pos_list[-1][1]}+{pos_list[0][1]}")
        return concat_pos(pos_list[1:], concat_pos_list)
    
    if len(concat_pos_list)>0 and concat_pos_list[-1][1].startswith(('NNG', 'NNP', 'XR')) and pos_list[0][1] in ['NNG', 'NNP', 'XR']:
        concat_pos_list[-1] = (concat_pos_list[-1][0]+pos_list[0][0],
                            f"{concat_pos_list[-1][1]}+{pos_list[0][1]}")
        return concat_pos(pos_list[1:], concat_pos_list)
    
    return concat_pos(pos_list[1:], concat_pos_list + [pos_list[0]])

def analyze_syntactics_with_mecab(sent : str, split_morphs : bool) -> tuple:
    pos_list = mecab.pos(sent)

    pos_list = concat_pos(pos_list, [])
    noun_list = [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR'))]
    
    verb_list = [pos for pos, tag in pos_list if tag.startswith(('XSV', 'VV', 'VA', 'VC'))]

    if not split_morphs:
        return [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR', 'XSV', 'VV', 'VA', 'VC'))]
    return noun_list, verb_list

def analyze_syntactics_with_okt(sent : str, split_morphs : bool) -> tuple:
    pos_list = okt.pos(sent)

    noun_list = [pos for pos, tag in pos_list if tag.startswith('Noun')]
    verb_list = [pos for pos, tag in pos_list if tag.startswith(('Adjective', 'Verb'))]

    if not split_morphs:
        return [pos for pos, tag in pos_list if tag.startswith(('Noun', 'Adjective', 'Verb'))]
    return noun_list, verb_list

def analyze_syntactics(sent : str, split_morphs=True, tokenizer='mecab') -> tuple:
    if tokenizer == 'mecab':
        return analyze_syntactics_with_mecab(sent, split_morphs)
    
    if tokenizer == 'okt':
        return analyze_syntactics_with_okt(sent, split_morphs)

def get_morphs(data : pd.DataFrame):
    data['query_pos'] = data['query'].apply(lambda x: ' '.join(analyze_syntactics(x, split_morphs=False)))
    return data