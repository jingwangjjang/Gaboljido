from model.model_ner.inference import make_candidates, return_dic

def run_ner(stt_list: list, ocr_list: list, region_code: int) -> dict:
    # 입력 리스트 그대로 전달 (내부에서 자동 처리됨)
    query_list = make_candidates(stt_list, ocr_list)
    
    # 최종 후보군에서 지역코드로 가장 유사한 매장 매칭
    result_dict = return_dic(region_code, query_list)
    
    return result_dict
