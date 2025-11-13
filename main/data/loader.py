from datasets import load_dataset

def load_csv_dataset(file_path, column_names=["text", "label"]):
    dataset = load_dataset(
        "csv",
        data_files=file_path,
        column_names=column_names,
    )["train"] # single file -> only "train" split

    return dataset

def load_many_csv_datasets(file_paths, column_names=["text", "label"]):
    dataset = load_dataset(
        "csv",
        data_files=file_paths,
        column_names=column_names,
    )

    return dataset
