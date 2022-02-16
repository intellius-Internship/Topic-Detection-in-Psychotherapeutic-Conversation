from kobart import get_kobart_tokenizer
from tokenization import KoBertTokenizer

from transformers import (ElectraForSequenceClassification, ElectraTokenizer, ElectraConfig,
                        BertForSequenceClassification, BertConfig,
                        AutoModelForSequenceClassification, AutoTokenizer, AutoConfig)

from transformers import (GPT2LMHeadModel, 
                        PreTrainedTokenizerFast,
                        BartForConditionalGeneration)


U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

def load_model(model_type, num_labels=0, labels=None, cache_dir='./cache'):
    if labels is None:
        labels = list(range(num_labels))

    if 'bert' == model_type:
        config = BertConfig.from_pretrained(
            "monologg/kobert",
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)}
        )
    
        model = BertForSequenceClassification.from_pretrained(
            "monologg/kobert", 
            config=config
        )
        tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        return model, tokenizer

    elif 'electra' == model_type:
        config = ElectraConfig.from_pretrained(
            "monologg/koelectra-base-v3-discriminator",
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)}
        )
        model = ElectraForSequenceClassification.from_pretrained(
            "monologg/koelectra-base-v3-discriminator", 
            config=config
        )
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        return model, tokenizer

    elif 'roberta' == model_type:
        config = AutoConfig.from_pretrained(
            "klue/roberta-base",
            num_labels=num_labels,
            cache_dir=cache_dir
        )
        config.label2id = {str(i): label for i, label in enumerate(labels)}
        config.id2label = {label: i for i, label in enumerate(labels)}
        model = AutoModelForSequenceClassification.from_pretrained(
            "klue/roberta-base", 
            config=config
        )
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        return model, tokenizer

    elif 'bigbird' == model_type:
        config = AutoConfig.from_pretrained("monologg/kobigbird-bert-base", 
                num_labels=num_labels,
                cache_dir=cache_dir)
        config.label2id = {str(i): label for i, label in enumerate(labels)}
        config.id2label = {label: i for i, label in enumerate(labels)}
        model = AutoModelForSequenceClassification.from_pretrained(
            "monologg/kobigbird-bert-base", 
            config=config
        )
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        return model, tokenizer

    elif 'bart' == model_type:
        model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        return model, tokenizer

    elif 'gpt2' == model_type:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

        return model, tokenizer
  
    raise NotImplementedError('Unknown model')


def load_tokenizer(model_name, cache_dir='./cache'):
    if 'bart' == model_name:
        return get_kobart_tokenizer(cache_dir)

    elif 'gpt2' == model_name:
        return PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

    raise NotImplementedError('Unknown model')