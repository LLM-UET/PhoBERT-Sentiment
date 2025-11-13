from ..data import (
    TRAIN_RATIO, CV_RATIO, TEST_RATIO,
    DEFAULT_SEGMENTED_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_CV_PATH,
    DEFAULT_TEST_PATH,
    load_csv_dataset,
)
import csv

def do_SPLIT(input_file_path=None, output_train_path=None, output_cv_path=None, output_test_path=None):
    ENCODING = 'utf-8-sig' # UTF-8 with BOM (EF BB BF)

    input_file_path = input_file_path or DEFAULT_SEGMENTED_PATH
    output_train_path = output_train_path or DEFAULT_TRAIN_PATH
    output_cv_path = output_cv_path or DEFAULT_CV_PATH
    output_test_path = output_test_path or DEFAULT_TEST_PATH

    assert abs((TRAIN_RATIO + CV_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    RANDOM_SEED = 42

    dataset = load_csv_dataset(input_file_path)

    # -----------------------------
    # Split train / cv / test dynamically
    # -----------------------------
    # First: all => train vs temp (cv+test)
    TEMP_RATIO = 1 - TRAIN_RATIO

    split1 = dataset.train_test_split(test_size=TEMP_RATIO, seed=RANDOM_SEED)
    train_dataset = split1["train"]
    temp_dataset = split1["test"]

    # Then: temp => cv vs test
    cv_ratio = CV_RATIO / (CV_RATIO + TEST_RATIO)
    split2 = temp_dataset.train_test_split(test_size=cv_ratio, seed=RANDOM_SEED)
    cv_dataset = split2["train"]
    test_dataset = split2["test"]

    # TODO: Save them to appropriate CSV files
    common_options = {
        "index": False,
        "encoding": ENCODING,
        "quoting": csv.QUOTE_ALL,
        "header": False,
    }
    train_dataset.to_csv(output_train_path, **common_options)
    cv_dataset.to_csv(output_cv_path, **common_options)
    test_dataset.to_csv(output_test_path, **common_options)

    total_count = len(dataset)
    total_count_after_splitting = len(train_dataset) + len(cv_dataset) + len(test_dataset)
    print(f"Total samples: {total_count}")
    print(f" - Train samples: {len(train_dataset)} ({len(train_dataset)/total_count:.2%})")
    print(f" - CV samples:    {len(cv_dataset)} ({len(cv_dataset)/total_count:.2%})")
    print(f" - Test samples:  {len(test_dataset)} ({len(test_dataset)/total_count:.2%})")

    lost_count = total_count - total_count_after_splitting
    print(f"Lost samples during splitting: {lost_count}")

    if lost_count == 0:
        print("Splitting completed successfully.")
    else:
        print("WARNING: Some samples were lost during splitting.")
