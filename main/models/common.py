import os

BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR", "./models/phobert-base-local")

FINETUNED_MODEL_DIR = os.getenv("FINETUNED_MODEL_DIR", "./models/finetuned")

LABELS = ['Negative', 'Neutral', 'Positive']

NUM_LABELS = len(LABELS)

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}
