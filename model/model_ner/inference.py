# Base
import re, ast, torch, os, time

# Load DB connection info from config
import yaml
import psycopg2

# Data
import numpy as np
import pandas as pd

# DB
import psycopg2
from sqlalchemy import create_engine

# Similarity
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from jamo.jamo import h2j
from Levenshtein import ratio
from konlpy.tag import Okt
okt = Okt()

# MODEL
from sentence_transformers import SentenceTransformer
# Load sentence transformer
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")
embedding_model.max_seq_length = 128

from transformers import AutoTokenizer, AutoModelForTokenClassification

from google.cloud import storage

def download_model_from_gcs(BUCKET_NAME, BLOB_DIR, LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=BLOB_DIR)
    for blob in blobs:
        if blob.name.endswith("/"):  # 폴더인 경우 스킵
            continue
        filename = blob.name.split("/")[-1]
        local_path = os.path.join(LOCAL_MODEL_DIR, filename)
        blob.download_to_filename(local_path)

BUCKET_NAME = "gaboljido_1"
BLOB_DIR = "ner_model/ner_output/"
LOCAL_MODEL_DIR = "./ner_output/"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/gynovzs/gabolgido-8f7f3309efa3.json"
download_model_from_gcs(BUCKET_NAME, BLOB_DIR, LOCAL_MODEL_DIR)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
    return tokenizer, model

ner_tokenizer, ner_model = load_model()
id2label = ner_model.config.id2label

# 텍스트 정규화
def normalize(text):
    text = text.lower()
    text = text.replace(" ", "")  # 공백 제거
    text = re.sub(r"[^가-힣a-z0-9]", "", text)  # 특수문자 제거
    return text

# 자모(Jamo) 유사도
def jamo_similarity(a, b):
    ja = ''.join(list(h2j(a)))
    jb = ''.join(list(h2j(b)))
    return ratio(ja, jb)

# 코사인 유사도
def cosine_sim(a, b):
    emb_a = embedding_model.encode([a], truncation=True, max_length=128)
    emb_b = embedding_model.encode([b], truncation=True, max_length=128)
    return float(cosine_similarity(emb_a, emb_b)[0][0])

# 최종 스마트 유사도
def smart_similarity(a, b, use_cosine=True):
    a_norm = normalize(a)
    b_norm = normalize(b)

    lev_score = ratio(a_norm, b_norm)
    jamo_score = jamo_similarity(a_norm, b_norm)
    cos_score = cosine_sim(a_norm, b_norm) if use_cosine else 0

    # 가중 평균 (필요시 조정)
    final_score = (lev_score * 0.4) + (jamo_score * 0.3) + (cos_score * 0.3)
    return final_score

def final_similarity(text1, text2, embedding1, embedding2, smart_weight=0.5, hybrid_weight=0.5):
    smart_sim = smart_similarity(text1, text2)
    hybrid_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    return smart_sim * smart_weight + hybrid_sim * hybrid_weight

# config.yaml
with open("/home/gynovzs/fastapi/model_server/config/config.yaml", "r") as file: 
    config = yaml.safe_load(file)
db_config = config["vectordb"]

# DB 연결
def vector_db_conn():
    return psycopg2.connect(
    host=db_config["host"],
    port=db_config["port"],
    user=db_config["user"],
    password=db_config["password"],
    dbname=db_config["name"])


def extract_place_candidates(text_list):

    place_candidates = []

    for text in text_list:
        norm_text = re.sub(r"[^가-힣a-zA-Z0-9 ]", "", text)  # 특수문자 제거
        norm_text = re.sub(r"(입니다|에요|했어요|있습니다|가세요|드세요|먹으세요)", "", norm_text)
        if len(norm_text) >= 2:
            place_candidates.append(norm_text) 

    return list(set(place_candidates))  # 중복 제거

def predict_store_names(text):
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = ner_model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    input_ids = inputs["input_ids"][0].tolist()

    tokens = ner_tokenizer.convert_ids_to_tokens(input_ids)
    id2label = ner_model.config.id2label

    store_names = []
    current_store = []

    for token, label_id in zip(tokens, predictions):
        label = id2label[label_id]

        # 예외 처리: special token 무시
        if token in [ner_tokenizer.cls_token, ner_tokenizer.sep_token, ner_tokenizer.pad_token]:
            continue

        clean_token = token.replace("▁", "").replace("##", "")

        if label == "B-상호":
            if current_store:
                store_names.append("".join(current_store))
                current_store = []
            current_store.append(clean_token)
        elif label == "I-상호" and current_store:
            current_store.append(clean_token)
        else:
            if current_store:
                store_names.append("".join(current_store))
                current_store = []

    if current_store:
        store_names.append("".join(current_store))

    return store_names

def get_loc_store(region_code):

    conn = vector_db_conn()
    cur = conn.cursor()
    cur.execute("""SELECT store_id, store_name, name_embedding, blog_reviews, visitor_reviews
                   FROM stores
                   WHERE region_code = {}""".format(region_code))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    columns = ['store_id', 'store_name', 'name_embedding', 'blog_reviews', 'visitor_reviews']
    df = pd.DataFrame(rows, columns=columns)

    df['total_reviews'] = df['blog_reviews'] + df['visitor_reviews']
    df.sort_values(by='total_reviews', ascending=False, inplace=True)

    return df


def make_candidates(stt_list, ocr_list):

    extracted_stt = extract_place_candidates(stt_list)
    extracted_ocr = extract_place_candidates(ocr_list)
    stt_query_list = []

    for word in list(set(extracted_stt)):

        tmp = predict_store_names(word)
        
        if len(tmp) != 0: stt_query_list.append(tmp[0])
    
    ocr_query_list = []

    for word in list(set(extracted_ocr)):

        tmp = predict_store_names(word)
        
        if len(tmp) != 0: ocr_query_list.append(tmp[0])

    return list(set(stt_query_list + ocr_query_list))


def return_dic(region_code, query_list):
    
    df = get_loc_store(region_code)
    print('지역코드 {} 매장 개수 {}개'.format(region_code, len(df)))  
    similar_dic = {}

    start_time = time.time()
    for query in query_list:
        
        if query in ["강남", "강동", "강북", "강서", "관악", "광진", "구로", "금천", "노원", "도봉", "동대문", "동작", "마포", "서대문", "서초", "성동", "성북", "송파", '양천', '영등포', '용산', "은평", "종로", "중구", "중랑", '성수', '연남']: continue

        query_embedding = embedding_model.encode(query, normalize_embeddings=True, truncation=True, max_length=128)
        similarities = []
        for _, row in df.iterrows():
            store_id = row['store_id']
            store_name = row['store_name']
            embedding = np.array(ast.literal_eval(row['name_embedding']))
            score = final_similarity(query, store_name, query_embedding, embedding)
        
            similarities.append((store_id, store_name, score))

        best_store_id, best_store_name, best_score = sorted(similarities, key=lambda x: x[2], reverse=True)[0]
        if best_score > 0.65:
            similar_dic[query] = [best_store_id, best_store_name, int(round(best_score, 2)*100)]

    # 하나도 출력 안됐을 경우, 해당 지역에서 리뷰 수 가장 많은 매장 5개 
    if len(similar_dic) == 0:
        
        df['total_reviews'] = df['blog_reviews'] + df['visitor_reviews']
        df.sort_values(by='total_reviews', ascending=False, inplace=True)

        for i in range(5):
            similar_dic[f'tmp_{i+1}'] = [int(df.iloc[i, 0]), df.iloc[i, 1], 100]
        print('유사 매장이 없어 해당 지역에서 리뷰 수가 가장 많은 5개 매장을 출력합니다.')
        return similar_dic
    
    # 중복 value 제거
    seen_pairs = set()
    filtered_dict = {}

    for key, value in similar_dic.items():
        dedup_key = (value[0], value[1])  # store_id와 store_name만 기준
        if dedup_key not in seen_pairs:
            seen_pairs.add(dedup_key)
            filtered_dict[key] = value
    # print('NER 결과 딕셔너리 개수: {}'.format(len(filtered_dict)))
    end_time = time.time()
    print(f'유사도 계산 소요 시간:{end_time - start_time:.2f}초')
    return filtered_dict
