import time
import sys

class Inferer:
    def __init__(self, input_model_dir: str | None = None, silent=False):
        import sys
        if silent:
            def LOG_INFO(msg: str):
                pass
        else:
            def LOG_INFO(msg: str):
                print(f"[INFO] {msg}", file=sys.stderr)
        self.LOG_INFO = LOG_INFO
    
        LOG_INFO(f"Importing required modules...")
        from ..models import get_tokenizer, get_classifier, FINETUNED_MODEL_DIR, ID_TO_LABEL

        input_model_dir = input_model_dir or FINETUNED_MODEL_DIR
        LOG_INFO(f"Loading models from: {input_model_dir}...")
        tokenizer = get_tokenizer(model_dir=input_model_dir)
        model = get_classifier(model_dir=input_model_dir)

        # tokenizer.eval()
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.ID_TO_LABEL = ID_TO_LABEL
        from ..finetuning import MAX_LEN
        self.MAX_LEN = MAX_LEN
        from ..SEGMENT import segment_text_directly
        self.segment_text_directly = segment_text_directly
    
    def segment(self, text: str):
        return self.segment_text_directly(text)
    
    def infer(self, text: str):
        self.LOG_INFO(f"Got input text: {text}")

        start_time = time.monotonic_ns()
        self.LOG_INFO(f"Segmenting input text...")
        segmented = self.segment_text_directly(text)
        self.LOG_INFO(f"Segmented input text: {segmented}")

        encoded = self.tokenizer(
            segmented,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.MAX_LEN,
        )

        import torch
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            print("LOGITS SHAPE:", logits.shape, file=sys.stderr)
            print("LOGITS:", logits, file=sys.stderr)
            pred_id = torch.argmax(logits, dim=-1).item()
        
        end_time = time.monotonic_ns()
        elapsed_ms = (end_time - start_time) / 1_000_000
        self.LOG_INFO(f"Inference completed in {elapsed_ms:.2f} ms")
        
        return self.ID_TO_LABEL[pred_id]


def do_INFER(text: str, input_model_dir=None, silent=False, interactive=False):
    inferer = Inferer(input_model_dir=input_model_dir, silent=silent)
    if not interactive:
        result = inferer.infer(text=text)
        return result
    else:
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("Input text: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Terminating.")
                break
            result = inferer.infer(text=user_input)
            print(f"Result: {result}")
        return ""
