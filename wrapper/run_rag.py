from model.model_rag_llm.utils import summarize_reviews  # 실제 요약 함수명 확인 필요

def run_rag(ner_result: dict) -> list:
    return summarize_reviews(ner_result)
