BASE_MODEL_DIR = "./models/phobert-base-local"

FINETUNED_MODEL_DIR = "./models/finetuned-phobert-sentiment"

LABELS = ['Positive', 'Neutral', 'Negative']

NUM_LABELS = len(LABELS)

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
