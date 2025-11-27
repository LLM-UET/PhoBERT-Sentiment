# PhoBERT Finetuning for Sentiment Classification

- [PhoBERT Finetuning for Sentiment Classification](#phobert-finetuning-for-sentiment-classification)
  - [Foreword](#foreword)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Prepare the Python environment](#prepare-the-python-environment)
    - [Downloading the Models](#downloading-the-models)
    - [Download RDRsegmenter JAR into project root](#download-rdrsegmenter-jar-into-project-root)
    - [Run the RDRsegmenter TXT-RPC server](#run-the-rdrsegmenter-txt-rpc-server)
    - [Setup Environment Variables](#setup-environment-variables)
  - [Prepare the Datasets](#prepare-the-datasets)
  - [Input Segmentation](#input-segmentation)
  - [Data Splitting](#data-splitting)
  - [Finetuning](#finetuning)
  - [Inference](#inference)
  - [References](#references)

## Foreword

**IF YOU JUST WANT INFERENCE**, just
follow the instructions in:

- [Setup](#setup)
- [Inference](#inference)

## Setup

### Prerequisites

- Python 3.12+
- Java 1.8+

### Prepare the Python environment

```sh
uv venv
source .venv/bin/activate
uv sync
```

### Downloading the Models

- **Base models** (not finetuned):

    ```sh
    source .venv/bin/activate
    uv run -m main download models BASE
    ```

    If you run the above commands
    twice, the models would not be
    downloaded again (i.e. it
    guarantees idempotency).

- **Finetuned models**:

    ```sh
    source .venv/bin/activate
    uv run -m main download models FINETUNED
    ```

    Beware, though, that this command is
    NOT idempotent. Running it the second
    time means re-downloading the models.

### Download RDRsegmenter JAR into project root
  
```sh
curl -L -O https://github.com/LLM-UET/RDRsegmenter/releases/download/0.9.1/RDRsegmenter.jar
```

Checksumming:

```sh
echo "47d3581e050bd686b9eac4727d4d453a5458d6b843e9b3c618a23f3654bdd7fa  RDRsegmenter.jar" | sha256sum --check
```

### Run the RDRsegmenter TXT-RPC server

This is required so that some stuff
could use RDRsegmenter via text RPC
(simple HTTP POST) to segment data,
as you would see later.

```sh
java -jar RDRsegmenter.jar TXT-RPC
```

or, to customize port:

```sh
java -jar RDRsegmenter.jar TXT-RPC 8025
```

### Setup Environment Variables

```sh
cp .env.example .env
```

then edit the variables in `.env`.

## Prepare the Datasets

- Datasets must be CSV files, **without headers**,
  consisting of `text,label` entries.

- It is highly recommended that the `text` part
  be wrapped in a pair of double quotes (`""`)
  to prevent incorrect parsing that stems from
  commas inside `text`.

- Example entries in a dataset file:

    ```csv
    "Nạp tiền mà sao chưa thấy vào tài khoản?",Negative
    "Gói Mimax70 có ổn không?",Neutral
    "Sao hủy gói F90 không được nhỉ",Negative
    "Data dùng tẹt ga luôn ^^",Positive
    ```

- Place a dataset file named `data.csv`
  under directory `<project_root>/datasets`.
  The app will split it into train/cv/test
  sets automatically and deterministically
  with a fixed seed.

- Before proceeding, **make sure the CSV file(s)**
  **do NOT contain "smart quotes"**, i.e. search
  for the following quotes: `“` and `”` and
  remove them both with normal double quotes `"`.

- Also watch out for open/unclosed/lone quotes,
  e.g. `"Have you finished this sentence yet,Negative`.

## Input Segmentation

If you've already done it with
`data.csv`, copy `data.csv` into a new
file named `segmented.csv`, under the
same directory.

Otherwise, run:

```sh
source .venv/bin/activate
uv run -m main segment
```

**It will output to `segmented.csv`** under the
same directory as `data.csv`.

**It should not be run more than once**
since `segment(segment(text)) != segment(text)`.

## Data Splitting

You must split `segmented.csv` into `train.csv`,
`cv.csv`, and `test.csv`.

To do that automatically:

```sh
source .venv/bin/activate
uv run -m main split
```

## Finetuning

```sh
source .venv/bin/activate
uv run -m main finetune
```

## Inference

```sh
source .venv/bin/activate
uv run -m main infer "text to infer"
```

**The input text must NOT be segmented.**

You can use `\n` to denote newlines inside
the input text string.

Output will be printed directly to `stdout`.

For interactive use (the downside: you
cannot insert newlines as input, even `\n` ;
pressing Enter means another query):

```sh
source .venv/bin/activate
uv run -m main infer --interactive
```

To change the model used for inference, pass
`--input-model-dir=/path/to/model`. Hint
(assuming you've [downloaded the necessary models](#downloading-the-models),
be it BASE, FINETUNED, or both):

- By default, it uses the FINETUNED model.
- To use BASE instead: pass the path `./models/phobert-base-local`.

## References

    @inproceedings{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020},
    pages     = {1037--1042}
    }
