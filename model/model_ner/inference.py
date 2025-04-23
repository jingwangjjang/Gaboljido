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
from rapidfuzz.distance import Levenshtein as RF_Levenshtein

# MODEL
from sentence_transformers import SentenceTransformer
# Load sentence transformer
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

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
        print(f"✅ 다운로드 완료: {filename}")

BUCKET_NAME = "gaboljido_1"
BLOB_DIR = "ner_model/ner_output/"
LOCAL_MODEL_DIR = "./ner_output/"

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/hojin/Downloads/gabolgido-8f7f3309efa3.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/gynovzs/gabolgido-8f7f3309efa3.json"

download_model_from_gcs(BUCKET_NAME, BLOB_DIR, LOCAL_MODEL_DIR)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
    return tokenizer, model

ner_tokenizer, ner_model = load_model()
id2label = ner_model.config.id2label

# Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# 유사도 계산
def get_embedding(text: str) -> list:
    if not text or pd.isna(text):
        return [0.0] * 768
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

def jaccard_similarity(a, b, n=2):
    ngrams_a = set([a[i:i+n] for i in range(len(a)-n+1)])
    ngrams_b = set([b[i:i+n] for i in range(len(b)-n+1)])
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b) if ngrams_a | ngrams_b else 0

def hybrid_score(query, candidate, query_emb, candidate_emb):
    emb_sim = cosine_similarity([query_emb], [candidate_emb])[0][0]
    edit_sim = fuzz.ratio(query, candidate) / 100
    jaccard_sim = jaccard_similarity(query, candidate)
    return 0.4 * emb_sim + 0.3 * edit_sim + 0.3 * jaccard_sim

def normalize(text):
    text = text.lower().replace(" ", "")
    return re.sub(r"[^가-힣a-z0-9]", "", text)

def jamo_similarity(a, b):
    ja = ''.join(list(h2j(a)))
    jb = ''.join(list(h2j(b)))
    return ratio(ja, jb)

def cosine_sim(a, b):
    emb_a = embedding_model.encode([a])
    emb_b = embedding_model.encode([b])
    return float(cosine_similarity(emb_a, emb_b)[0][0])

def refined_lev_score(a, b):
    if not a or not b:
        return 0.0
    dist = RF_Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return 1 - dist / max_len

def smart_similarity(a, b, use_cosine=True):
    a_norm = normalize(a)
    b_norm = normalize(b)
    lev_score = refined_lev_score(a_norm, b_norm)
    jamo_score = jamo_similarity(a_norm, b_norm)
    cos_score = cosine_sim(a_norm, b_norm) if use_cosine else 0
    return (lev_score * 0.1) + (jamo_score * 0.45) + (cos_score * 0.45)

def final_similarity(text1, text2, embedding1, embedding2, smart_weight=0.5, hybrid_weight=0.5):
    smart_sim = smart_similarity(text1, text2)
    hybrid_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    return smart_sim * smart_weight + hybrid_sim * hybrid_weight


# config.yaml
#with open("../../config/config.yaml", "r") as file:
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

def predict_store_names(text):

    tokens = ner_tokenizer(text.split(), return_tensors="pt", is_split_into_words=True, truncation=True)
    with torch.no_grad():
        outputs = ner_model(**tokens)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    input_ids = tokens["input_ids"].squeeze().tolist()
    word_ids = tokens.word_ids()

    id2label = ner_model.config.id2label
    store_names = []
    current_store = ""

    prev_label = "O"
    prev_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue

        token = ner_tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
        label = id2label[predictions[i]]

        if label == "B-상호":
            if current_store:
                store_names.append(current_store)
            current_store = text.split()[word_id]
        elif label == "I-상호" and current_store:
            current_store += text.split()[word_id]
        else:
            if current_store:
                store_names.append(current_store)
                current_store = ""

        prev_word_id = word_id

    if current_store:
        store_names.append(current_store)

    return store_names

def get_loc_store(region_code):

    conn = vector_db_conn()
    cur = conn.cursor()
    cur.execute("""SELECT store_id, store_name, name_embedding
                   FROM stores
                   WHERE region_code = {}""".format(region_code))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

def make_candidates(stt_list, ocr_list):

    combined = list(set(stt_list + ocr_list))

    query_list = []

    for word in combined:

        tmp = predict_store_names(word)
        
        if len(tmp) != 0: query_list.append(tmp[0])

    return query_list

def return_dic(region_code, query_list):

    similar_dic = {}

    rows = get_loc_store(region_code)
    print('지역코드 {} 매장 개수'.format(len(rows)))
    start_time = time.time()
    for query in query_list:

        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        similarities = []
        for store_id, store_name, embedding in rows:
            score = final_similarity(query, store_name, query_embedding, np.array(ast.literal_eval(embedding)))
            similarities.append((store_id, store_name, score))

        best_store_id, best_store_name, best_score = sorted(similarities, key=lambda x: x[2], reverse=True)[0]
        if best_score > 0.55:
            similar_dic[query] = [best_store_id, best_store_name, int(round(best_score, 2)*100)]

    end_time = time.time()
    print(f"⏱️ 유사도 계산 실행 시간: {end_time - start_time:.4f}초")

    # if len(similar_dic) == 0:

    #     # 쿼리 날려서 해당 지역의 매장 리뷰 수 많은 거라도???
    print('NER 결과 딕셔너리 개수: {}'.format(len(similar_dic)))
    print(similar_dic) 
    return similar_dic

