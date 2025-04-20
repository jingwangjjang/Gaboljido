from model.model_rag_llm.utils import generate_store_summaries  

def run_rag(ner_result: dict) -> dict:
    return generate_store_summaries(ner_result)
