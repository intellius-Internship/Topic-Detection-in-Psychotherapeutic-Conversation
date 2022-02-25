import pandas as pd

from typing import List
from konlpy.tag import Okt 
from konlpy.tag import Mecab
from tqdm.auto import tqdm

mecab = Mecab()
okt = Okt()


'''
Description
-----------
어말 어미와 동사/형용사와 어말 어미를 연결 
또는 복합 명사 연결
'''
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

'''
Description
-----------
Mecab tokenizer를 이용한 형태소 분석
'''
def analyze_syntactics_with_mecab(sent : str, split_morphs : bool) -> tuple:
    pos_list = mecab.pos(sent)

    pos_list = concat_pos(pos_list, [])
    noun_list = [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR'))]
    verb_list = [pos for pos, tag in pos_list if tag.startswith(('XSV', 'VV', 'VA', 'VC'))]

    if not split_morphs:
        return [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR', 'XSV', 'VV', 'VA', 'VC'))]
    return noun_list, verb_list


'''
Description
-----------
Okt tokenizer를 이용한 형태소 분석
'''
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

'''
Description
-----------
사용자 발화에 대한 형태소 분석 수행 
명사, 형용사, 동사를 제외한 품사 제거
'''
def get_morphs(data : pd.DataFrame, tokenizer : str):
    tqdm.pandas(desc="split_morphs")
    data['query_pos'] = data['query'].progress_apply(lambda x: ' '.join(analyze_syntactics(x, \
        split_morphs=False, tokenizer=tokenizer)))
    return data