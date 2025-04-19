import os
import re
import ast
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load DB connection info from .env
vdb_user = os.getenv("VDB_USER")
vdb_password = os.getenv("VDB_PASSWORD")
vdb_host = os.getenv("VDB_HOST")
vdb_port = int(os.getenv("VDB_PORT", "5432"))
vdb_name = os.getenv("VDB_NAME")

vdb_engine = create_engine(f"postgresql+psycopg2://{vdb_user}:{vdb_password}@{vdb_host}:{vdb_port}/{vdb_name}")

# psycopg2 connection
def vector_db_conn():
    return psycopg2.connect(
        host=vdb_host,
        port=vdb_port,
        dbname=vdb_name,
        user=vdb_user,
        password=vdb_password
    )

# Load sentence transformer
model = SentenceTransformer("jhgan/ko-sbert-nli")

def get_embedding(text: str) -> list:
    if not text or pd.isna(text):
        return [0.0] * 768
    return model.encode(text, convert_to_numpy=True).tolist()

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
    emb_a = model.encode([a])
    emb_b = model.encode([b])
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

def generate_store_summary(query: str):
    try:
        # 후보 store들 불러오기
        conn = vector_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT store_id, store_name, name_embedding FROM stores")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        query_embedding = model.encode(query, normalize_embeddings=True)
        similarities = []
        for store_id, store_name, embedding in rows:
            score = final_similarity(query, store_name, query_embedding, np.array(ast.literal_eval(embedding)))
            similarities.append((store_id, store_name, score))

        best_store_id, best_store_name, best_score = sorted(similarities, key=lambda x: x[2], reverse=True)[0]

        # 해당 매장의 리뷰 불러오기
        conn = vector_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT review_docs FROM stores WHERE store_id = %s", (best_store_id,))
        review_rows = cur.fetchall()
        cur.close()
        conn.close()

        reviews = [review[0] for review in review_rows if review[0]]
        while len(reviews) < 7:
            reviews.append("(리뷰 없음)")

        formatted_reviews = '\n'.join([f" -- {r}" for r in reviews])

        # LLM 프롬프트 생성
        system_prompt = f"""
        ## Persona
        당신은 '매장명', '매장리뷰'를 바탕으로 매장에 대한 정보를 3줄(100자 이내)로 답변하는 3명의 리뷰어입니다.

        ## Instruction
        - 어떠한 일이 있어도 반드시 친근한 말투로 3줄(100자 이내)로 답변해야 한다.
        - 모든 답변은 반드시 '한국어'로 해야한다.
        - 문장은 완결형으로 자연스럽게 마무리합니다.
        - 각 줄은 마치 실제 사람이 말하듯 자연스럽고 생동감 있는 톤으로 답변합니다.
        - 만약, 매장리뷰가 없다면 '해당하는 매장 리뷰가 존재하지 않습니다.' 로 답변합니다.
        - 오로지 '매장리뷰'에 있는 정보만을 활용해서 답변합니다.

        ## Information
        - **매장명**: {best_store_name}
        - **매장리뷰**:
        {formatted_reviews}

        ## Example
        USER: 매장 정보 제공 요청
        ASSISTANT: 너무 좋았어요! 다음에 친구들이랑 한 번 더 오려구요!
        날씨 좋은 날에 처음 가봤는데, 대박 좋았어요~!
        매장이 깔끔해서 좋고, 사장님이 엄청 친절해요~
        USER: 매장 정보 제공 요청
        ASSISTANT: 해당하는 매장 리뷰가 존재하지 않습니다.
        해당하는 매장 리뷰가 존재하지 않습니다.
        해당하는 매장 리뷰가 존재하지 않습니다.
        """

        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        user_prompt = "매장 정보 제공 요청"

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        response = completion.choices[0].message.content.strip().split('\n')[:3]

        return {
            "status": "success",
            "code": 200,
            "message": "RAG LLM 요청 완료",
            "processed": "ok",
            "data": {
                "review_1": response[0] if len(response) > 0 else "",
                "review_2": response[1] if len(response) > 1 else "",
                "review_3": response[2] if len(response) > 2 else "",
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": str(e),
            "processed": "failed",
            "data": {}
        }


# # 예시 실행
# if __name__ == "__main__":
#     result = generate_store_summary("가볼지도")
#     print(result)
