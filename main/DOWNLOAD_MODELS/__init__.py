def do_DOWNLOAD_MODELS_BASE():
    from ..models import get_tokenizer, get_classifier

    tokenizer = get_tokenizer(for_download=True)
    get_classifier = get_classifier(for_download=True)

    return (tokenizer, get_classifier)

def do_DOWNLOAD_MODELS_FINETUNED():
    from ..models import FINETUNED_MODEL_DIR
    raise NotImplementedError("TODO: This will be implemented soon")
