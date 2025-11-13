def do_DOWNLOAD_MODELS_BASE():
    from ..models import get_tokenizer, get_classifier

    tokenizer = get_tokenizer() # (force_download=True)
    get_classifier = get_classifier() # (force_download=True)

    return (tokenizer, get_classifier)

def do_DOWNLOAD_MODELS_FINETUNED():
    raise NotImplementedError("TODO: This will be implemented soon")
