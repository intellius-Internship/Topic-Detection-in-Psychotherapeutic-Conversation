# **Topic-Detection in Psychotherapeutic Conversation**


### **파일 구조**

```bash
.
├── data    
│   ├── kw_based/               키워드 기반 labeled conversation dataset
│   ├── regexp_based/           정규식 매칭 기반 labeled conversation dataset
│   │   
│   ├── DATA.md                
│   ├── hypothesis.csv          Raw Conversation Dataset (약 300만건의 싱글턴 대화)
│   ├── reference.csv           웰니스 심리상담 대화 데이터 (약 1만 9천건)
│   └── topic.csv               Topic-KW-RegExp Dataset, 19개 상담주제 클래스의 키워드/정규식
│
├── preprocessing               데이터 라벨링 및 전처리 
│   ├── ...
│   ├── build_dataset.py        데이터셋 구축을 위한 실행 코드
│   └── ...                 
│
├── result/                     모델 테스트 결과 저장 경로
├── utils/
├── ...
├── main.py                     모델 학습 및 테스트를 위한 실행 코드
├── READMD.md
└── ...
```

<br>


## **Building Topic-Detection Dataset** 


```bash
cd preprocessing/
```

### 1. Topic Labeling

- `labeling`: 라벨링 메소드 (default=None)   
    - `keyword` : 키워드 기반 라벨링
    - `textdist` : 텍스트 유사도 기반 라벨링
    - `vector` : tfidf vectorization 기반 라벨링
    - `regexp` : 정규식 매칭 기반 라벨링

***주의***  
`textdist`와 `vector`는 약 2만건의 웰니스 대화 데이터와 300만건의 일상 대화 간 비교를 수행하므로 실행 시간이 오래 걸림


```bash
python build_dataset.py --labeling regexp --data_dir ../data --result_dir ../result
```

### 2. Build Training, Validation, Test dataset
```bash
python build_dataset.py --preprocessing --split --data_dir ../data --result_dir ../result
```

<br>

---

## **Training/Testing Topic Detection Model** 

<br>

- `model_type`: 모델 유형      
    - `gpt2` : Pretrained KoGPT2 (`skt/kogpt2-base-v2`)
    - `bart` : Pretrained KoBART (`gogamza/kobart-base-v2`)
    - `bert` : Pretrained KoBERT (`monologg/kobert`)
    - `electra` : Pretrained KoELECTRA (`monologg/koelectra-base-v3-discriminator`)
    - `bigbird` : Pretrained KoBigBird (`monologg/kobigbird-bert-base`)
    - `roberta` : Pretrained KoRoBERTa (`klue/roberta-base`)

### 1. Training

```bash
python main.py --train --max_epochs 10 --data_dir data/regexp_based --model_type roberta --model_name roberta+regexp --max_len 64 --gpuid 0
```

<br>

### 2. Testing

*하나의 GPU만 사용*  

#### (1) `<data_dir>`/test.csv에 대한 성능 테스트

```bash
python main.py --data_dir data/regexp_based --model_type roberta --model_name roberta+regexp --save_dir result --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

#### (2) 사용자 입력에 대한 성능 테스트

```bash
python main.py --user_input --data_dir data/regexp_based --model_type roberta --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

<br>


