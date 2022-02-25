
# -*- coding: utf-8 -*-
import re
import torch
import pandas as pd
from os.path import join as pjoin
from utils.data_util import encode

from plm import LightningPLM
from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel

from utils.model_util import U_TKN, S_TKN
from dataloader import DELIMITER

'''
Description
-----------
사용자 입력이 유효한지 판단
'''
def is_valid(query):
    if not re.sub('[\s]+', '', query):
        return False
    return True

'''
Description
-----------
GPT2 기반 대화 주제 탐지 모델 test data에서의 테스트
'''
def eval_ar(args, model, tokenizer, device, test_data):
    u_tkn, s_tkn = U_TKN, S_TKN

    topic_list = []
    acc = 0
    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']
            tgt_topic = row['label_str']

            # encodinig user utterance
            q_toked = tokenizer.tokenize(u_tkn + query)
            if len(q_toked) >= args.max_len:
                q_toked = [q_toked[0]] + q_toked[-(int(args.max_len/2))+1:]

            topic = ''
            # inference
            for iter_ in range(args.max_len):
                r_toked = tokenizer.tokenize(s_tkn + topic)
                token_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(q_toked + r_toked)).to(device=device)

                logits = model(token_ids)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                topic += gen.replace('▁', ' ')

            topic= topic.strip()
            topic_list.append(topic)
            print("Topic: {}".format(topic))
            
            if topic == tgt_topic:
                acc += 1

        # save test result to <save_dir>
        test_data['pred_topic'] = topic_list
        print("Accuracy: {}".format(acc / len(test_data)))
        test_data.to_csv(f'{args.save_dir}/{args.model_name}-{round(acc/len(test_data), 2)*100}.csv', index=False)

'''
Description
-----------
BART 기반 대화 주제 탐지 모델 test data에서의 테스트
'''
def eval_s2s(args, model, tokenizer, device, test_data):
    topic_list = []
    acc = 0

    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']
            tgt_topic = row['label_str']

            # encodinig user utterance
            enc_input, attention_mask = encode(tokenizer=tokenizer, \
                sent=tokenizer.bos_token+query+tokenizer.eos_token, \
                max_len=args.max_len)

            enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device=device)
            attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

            topic = ''
            # inference
            for iter_ in range(args.max_len-1):
                dec_input, dec_attention_mask = encode(tokenizer=tokenizer, \
                    sent=tokenizer.bos_token+topic, max_len=args.max_len)

                dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device=device)
                dec_attention_mask = torch.FloatTensor(dec_attention_mask).unsqueeze(0).to(device=device)
    
                inputs = {
                    "input_ids": enc_input,
                    "attention_mask" : attention_mask,
                    "decoder_input_ids" : dec_input,
                    "decoder_attention_mask" : dec_attention_mask,
                    "labels": None
                }
                outs = model(inputs)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break

                topic += gen.replace('▁', ' ')

            topic= topic.strip()
            topic_list.append(topic)
            print("Topic: {}".format(topic))

            if topic == tgt_topic:
                acc += 1

        # save test result to <save_dir>
        test_data['pred_topic'] = topic_list
        print("Accuracy: {}".format(acc / len(test_data)))

        test_data.to_csv(f'{args.save_dir}/{args.model_name}-{round(acc/len(test_data), 2)*100}.csv', index=False)

'''
Description
-----------
Transformer 기반 대화 주제 탐지 모델 test data에서의 테스트
'''
def eval_transformer(args, model, tokenizer, device, test_data):
    pred_list = []
    count = 0
    with torch.no_grad():
        for row in test_data.iterrows():
            text = row[1]['proc_query']
            label = int(row[1]['label'])
            assert isinstance(text, str)
            
            # encodinig user utterance
            input_ids, _ = encode(tokenizer.cls_token + text \
                +tokenizer.sep_token, tokenizer=tokenizer, max_len=args.max_len)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = None

            # inference
            logits = model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            pred_list.append(predictions[0]) 

            if predictions[0] == label:
                count += 1

        # save test result to <save_dir>
        test_data['pred'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'{args.model_name}-{round(count/len(test_data), 2)*100}.csv'), sep='\t', index=False)
        print(f"Accuracy: {count/len(test_data)}")            

'''
Description
-----------
GPT2 기반 대화 주제 탐지 모델 사용자 입력에 대한 테스트
'''
def eval_ar_usr_input(args, model, tokenizer, device):
    u_tkn, s_tkn = U_TKN, S_TKN

    with torch.no_grad():
        u_utter = input("User Utterance: ")
        while is_valid(u_utter):

            topic = ''
            # encodinig user utterance
            q_toked = tokenizer.tokenize(u_tkn + u_utter)
            if len(q_toked) >= args.max_len:
                q_toked = [q_toked[0]] + q_toked[-(int(args.max_len/2))+1:]

            # inference
            for iter_ in range(args.max_len):
                r_toked = tokenizer.tokenize(s_tkn + topic)
                token_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(q_toked + r_toked)).to(device=device)

                logits = model(token_ids)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                topic += gen.replace('▁', ' ')

            topic= topic.strip()
            print("Topic: {}".format(topic)) 

            u_utter = input("User Utterance: ")  

'''
Description
-----------
BART 기반 대화 주제 탐지 모델 사용자 입력에 대한 테스트
'''            
def eval_s2s_usr_input(args, model, tokenizer, device):
    with torch.no_grad():
        u_utter = input("User Utterance: ")
        while is_valid(u_utter):
            # encodinig user utterance
            enc_input, attention_mask = encode(tokenizer=tokenizer, \
                sent=tokenizer.bos_token+u_utter+tokenizer.eos_token, \
                max_len=args.max_len)

            enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device=device)
            attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

            topic = ''
            # inference
            for iter_ in range(args.max_len-1):
                dec_input, dec_attention_mask = encode(tokenizer=tokenizer, \
                    sent=tokenizer.bos_token+topic, max_len=args.max_len)

                dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device=device)
                dec_attention_mask = torch.FloatTensor(dec_attention_mask).unsqueeze(0).to(device=device)
    
                inputs = {
                    "input_ids": enc_input,
                    "attention_mask" : attention_mask,
                    "decoder_input_ids" : dec_input,
                    "decoder_attention_mask" : dec_attention_mask,
                    "labels": None
                }
                outs = model(inputs)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                topic += gen.replace('▁', ' ')

            topic= topic.strip()
            print("Topic: {}".format(topic))

            u_utter = input("User Utterance: ")

'''
Description
-----------
Transformer 기반 대화 주제 탐지 모델 사용자 입력에 대한 테스트
'''
def eval_tf_usr_input(args, model, tokenizer, device, labels):
    with torch.no_grad():
        
        u_utter = input("User Utterance: ")
        while is_valid(u_utter):

            # encoding user utterance
            input_ids, _ = encode(tokenizer.cls_token + u_utter + tokenizer.sep_token, \
                tokenizer=tokenizer, max_len=args.max_len)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = None

            # inference
            logits = model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            
            print(f"Topic: {predictions} ({labels[labels.label==predictions[0]].label_str.tolist()[0]})")

            u_utter = input("User Utterance: ")

def evaluation(args, **kwargs):
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    # load model checkpoint
    if args.model_pt is not None:
        if args.model_type in ['bert', 'electra', 'bigbird', 'roberta']:
            model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
        elif args.model_type in ['gpt2']:
            model = AutoRegressiveModel.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))
        else:
            model = Seq2SeqModel.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))

    # freeze model params
    model = model.cuda()     
    model.eval()

    # load test dataset
    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    test_data.dropna(axis=0, inplace=True)

    labels = test_data[['label_str', 'label']]

    # if model_type is transformer-based
    if args.model_type in ['bert', 'electra', 'bigbird', 'roberta']:
        if args.user_input:
            eval_tf_usr_input(args, model=model, tokenizer=model.tokenizer, device=device, labels=labels)
        else:
            eval_transformer(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)
    
    # if model_type is autoregressive model
    elif args.model_type in ['gpt2']:
        if args.user_input:
            eval_ar_usr_input(args, model=model, tokenizer=model.tokenizer, device=device)
        else:
            eval_ar(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)
    
    # if model_type is seq2seq model
    else:
        if args.user_input:
            eval_s2s_usr_input(args, model=model, tokenizer=model.tokenizer, device=device)
        else:
            eval_s2s(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)

            

