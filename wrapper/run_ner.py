from model.model_ner.inference import make_candidates, return_dic

def run_ner(stt_list: list, ocr_list: list, region_code: int) -> dict:
    query_list = make_candidates(stt_list, ocr_list)
    return return_dic(region_code, query_list)
