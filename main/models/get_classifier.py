from .common import BASE_MODEL_DIR, NUM_LABELS

def get_classifier(model_dir=None, for_download=False):
    from transformers import AutoModelForSequenceClassification

    if model_dir is None:
        model_dir = BASE_MODEL_DIR

    # if for_download:
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         "vinai/phobert-base",
    #         num_labels=NUM_LABELS,
    #         cache_dir=BASE_MODEL_DIR,
    #     )
    # else:
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=NUM_LABELS,
        cache_dir=model_dir,
    )
    
    return model
