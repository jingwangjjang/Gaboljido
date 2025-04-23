import os
import re
import json
import ast
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from groq import Groq
from dotenv import load_dotenv

# 환경 설정
load_dotenv()

vdb_user = os.getenv("VDB_USER")
vdb_password = os.getenv("VDB_PASSWORD")
vdb_host = os.getenv("VDB_HOST")
vdb_port = int(os.getenv("VDB_PORT", "5432"))
vdb_name = os.getenv("VDB_NAME")
vdb_engine = create_engine(f"postgresql+psycopg2://{vdb_user}:{vdb_password}@{vdb_host}:{vdb_port}/{vdb_name}")

def vector_db_conn():
    return psycopg2.connect(
        host=vdb_host,
        port=vdb_port,
        dbname=vdb_name,
        user=vdb_user,
        password=vdb_password
    )




# 문장 자동 분리 유틸 함수
def split_reviews(response_text):
    sentences = re.split(r'(?<=[.!?~])\s+', response_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = sentences[:3]
    while len(sentences) < 3:
        sentences.append(None)
    return {
        "review_1": sentences[0],
        "review_2": sentences[1],
        "review_3": sentences[2],
    }

# 리뷰 요약 생성 함수
def generate_store_summaries(stt_results: dict):
    try:
        summaries = []
        conn = vector_db_conn()
        cur = conn.cursor()

        for keyword, (store_id, store_name, confidence) in stt_results.items():
            print(f"\n📌 [진행 중] 매장 처리 시작 - '{store_name}' (ID: {store_id}, 키워드: '{keyword}', 신뢰도: {confidence}%)")

            # 해당 store_id에 대한 리뷰 조회
            cur.execute("SELECT review_docs FROM stores WHERE store_id = %s", (store_id,))
            review_rows = cur.fetchall()
            reviews = [row[0] for row in review_rows if row[0]]

            print(f"🔍 [리뷰 조회 완료] '{store_name}'의 리뷰 {len(reviews)}개 조회됨")

            # 리뷰 7개 이하일 경우 패딩
            while len(reviews) < 7:
                reviews.append("(리뷰 없음)")

            formatted_reviews = '\n'.join([f" -- {r}" for r in reviews])
            print(f"📝 [프롬프트 생성] '{store_name}' 리뷰 형식화 완료")

            # LLM 프롬프트 구성
            system_prompt = f"""
            ## Persona
            당신은 '매장명', '매장리뷰'를 바탕으로 매장에 대한 정보를 3줄(100자 이내)로 답변하는 3명의 리뷰어입니다.

            ## Instruction
            - 모든 문장은 '한국어'로 작성하고, 3줄(100자 이내)의 완결형 문장으로 작성합니다.
            - 반드시 '방문자'의 시점에서 직접 경험한 것처럼 작성하세요.
            - 각 줄은 마치 실제 사람이 말하듯 자연스럽고 생동감 있는 톤으로 답변합니다.
            - 리뷰가 없다면, 3줄 모두 "해당하는 매장 리뷰가 존재하지 않습니다."로 작성합니다.
            - 오로지 '매장리뷰'에 있는 정보만을 활용해서 답변합니다.

            ## Information
            - **매장명**: {store_name}
            - **매장리뷰**:
            {formatted_reviews}

            ## Example
            USER: 매장 정보 제공 요청
            ASSISTANT:
            너무 좋았어요! 다음에 친구들이랑 한 번 더 오려구요!
            날씨 좋은 날에 처음 가봤는데, 대박 좋았어요~!
            매장이 깔끔해서 좋고, 사장님이 엄청 친절해요~            
            USER: 매장 정보 제공 요청
            ASSISTANT: 
            해당하는 매장 리뷰가 존재하지 않습니다.
            해당하는 매장 리뷰가 존재하지 않습니다.
            해당하는 매장 리뷰가 존재하지 않습니다.
            """

            # Groq LLM 호출
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
            response_raw = completion.choices[0].message.content.strip().split('\n')
            response = [r.strip() for r in response_raw if r.strip()]

            # 응답이 1줄인데 문장 3개가 포함된 경우 분리
            if len(response) == 1:
                review_parts = split_reviews(response[0])
            else:
                review_parts = {
                    "review_1": response[0] if len(response) > 0 else None,
                    "review_2": response[1] if len(response) > 1 else None,
                    "review_3": response[2] if len(response) > 2 else None,
                }

            print(f"✅ [LLM 응답 완료] '{store_name}' 리뷰 요약 생성 완료")

            summaries.append({
                "keyword": keyword,
                "store_id": store_id,
                "store_name": store_name,
                "confidence": confidence,
                "review_1": review_parts["review_1"],
                "review_2": review_parts["review_2"],
                "review_3": review_parts["review_3"],
            })

        cur.close()
        conn.close()

        # 최종 결과 로그 확인
        print("\n🎉 [전체 완료] 모든 매장 요약 작업이 성공적으로 완료되었습니다.")
        print(f"\n🧾 [요약 결과 데이터]:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}")  # ✅ 결과 로그 출력

# 문장 자동 분리 유틸 함수
def split_reviews(response_text):
    sentences = re.split(r'(?<=[.!?~])\s+', response_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = sentences[:3]
    while len(sentences) < 3:
        sentences.append(None)
    return {
        "review_1": sentences[0],
        "review_2": sentences[1],
        "review_3": sentences[2],
    }

# 리뷰 요약 생성 함수
def generate_store_summaries(stt_results: dict):
    try:
        summaries = []
        conn = vector_db_conn()
        cur = conn.cursor()

        for keyword, (store_id, store_name, confidence) in stt_results.items():
            print(f"\n📌 [진행 중] 매장 처리 시작 - '{store_name}' (ID: {store_id}, 키워드: '{keyword}', 신뢰도: {confidence}%)")

            # 해당 store_id에 대한 리뷰 조회
            cur.execute("SELECT review_docs FROM stores WHERE store_id = %s", (store_id,))
            review_rows = cur.fetchall()
            reviews = [row[0] for row in review_rows if row[0]]

            print(f"🔍 [리뷰 조회 완료] '{store_name}'의 리뷰 {len(reviews)}개 조회됨")

            # 리뷰 7개 이하일 경우 패딩
            while len(reviews) < 7:
                reviews.append("(리뷰 없음)")

            formatted_reviews = '\n'.join([f" -- {r}" for r in reviews])
            print(f"📝 [프롬프트 생성] '{store_name}' 리뷰 형식화 완료")

            # LLM 프롬프트 구성
            system_prompt = f"""
            ## Persona
            당신은 '매장명', '매장리뷰'를 바탕으로 매장에 대한 정보를 3줄(100자 이내)로 답변하는 3명의 리뷰어입니다.

            ## Instruction
            - 모든 문장은 '한국어'로 작성하고, 3줄(100자 이내)의 완결형 문장으로 작성합니다.
            - 반드시 '방문자'의 시점에서 직접 경험한 것처럼 작성하세요.
            - 각 줄은 마치 실제 사람이 말하듯 자연스럽고 생동감 있는 톤으로 답변합니다.
            - 리뷰가 없다면, 3줄 모두 "해당하는 매장 리뷰가 존재하지 않습니다."로 작성합니다.
            - 오로지 '매장리뷰'에 있는 정보만을 활용해서 답변합니다.

            ## Information
            - **매장명**: {store_name}
            - **매장리뷰**:
            {formatted_reviews}

            ## Example
            USER: 매장 정보 제공 요청
            ASSISTANT:
            너무 좋았어요! 다음에 친구들이랑 한 번 더 오려구요!
            날씨 좋은 날에 처음 가봤는데, 대박 좋았어요~!
            매장이 깔끔해서 좋고, 사장님이 엄청 친절해요~            
            USER: 매장 정보 제공 요청
            ASSISTANT: 
            해당하는 매장 리뷰가 존재하지 않습니다.
            해당하는 매장 리뷰가 존재하지 않습니다.
            해당하는 매장 리뷰가 존재하지 않습니다.
            """

            # Groq LLM 호출
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
            response_raw = completion.choices[0].message.content.strip().split('\n')
            response = [r.strip() for r in response_raw if r.strip()]

            # 응답이 1줄인데 문장 3개가 포함된 경우 분리
            if len(response) == 1:
                review_parts = split_reviews(response[0])
            else:
                review_parts = {
                    "review_1": response[0] if len(response) > 0 else None,
                    "review_2": response[1] if len(response) > 1 else None,
                    "review_3": response[2] if len(response) > 2 else None,
                }

            print(f"✅ [LLM 응답 완료] '{store_name}' 리뷰 요약 생성 완료")

            summaries.append({
                "keyword": keyword,
                "store_id": store_id,
                "store_name": store_name,
                "confidence": confidence,
                "review_1": review_parts["review_1"],
                "review_2": review_parts["review_2"],
                "review_3": review_parts["review_3"],
            })

        cur.close()
        conn.close()

        print("\n🎉 [전체 완료] 모든 매장 요약 작업이 성공적으로 완료되었습니다.")

        return {
            "status": "success",
            "code": 200,
            "message": "모든 매장 요약 완료",
            "processed": "ok",
            "data": summaries
        }

    except Exception as e:
        print(f"\n❌ [오류 발생] {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "message": str(e),
            "processed": "failed",
            "data": []
        }


