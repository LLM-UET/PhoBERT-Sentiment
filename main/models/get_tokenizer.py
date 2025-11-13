from .common import BASE_MODEL_DIR

def get_tokenizer(model_dir=None, for_download=False):
    from transformers import AutoTokenizer

    if model_dir is None:
        model_dir = BASE_MODEL_DIR

    # if for_download:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "vinai/phobert-base",
    #         cache_dir=BASE_MODEL_DIR,
    #     )
    # else:
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base",
        cache_dir=model_dir,
    )
    
    return tokenizer
