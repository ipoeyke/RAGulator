# RAGulator

This repository contains code to perform out-of-context detection with RAGulator, as described in [**RAGulator: Lightweight Out-of-Context Detectors for Grounded Text Generation**](https://arxiv.org/abs/2411.03920).

## Key Points
* RAGulator predicts whether a sentence is out-of-context (OOC) from retrieved text documents in a RAG setting.
* We preprocess a combination of summarisation and semantic textual similarity datasets (STS) to construct training data using minimal resources.
* We demonstrate 2 types of trained models: tree-based meta-models trained on features engineered on preprocessed text, and BERT-based classifiers fine-tuned directly on original text.
* We find that fine-tuned DeBERTa is not only the best-performing model under this pipeline, but it is also fast and does not require additional text preprocessing or feature engineering.

## Model Details

### Dataset
Training data for RAGulator is adapted from a combination of summarisation and STS datasets to simulate RAG:
* [BBC](https://www.kaggle.com/datasets/pariza/bbc-news-summary)
* [CNN DailyMail ver. 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail)
* [PubMed](https://huggingface.co/datasets/ccdv/pubmed-summarization)
* [MRPC from the GLUE dataset](https://huggingface.co/datasets/nyu-mll/glue/)
* [SNLI ver. 1.0](https://huggingface.co/datasets/stanfordnlp/snli)

The datasets were transformed before concatenation into the final dataset. Each row of the final dataset consists \[`sentence`, `context`, `OOC label`\].
* For summarisation datasets, transformation was done by randomly pairing summary abstracts with unrelated articles to create OOC pairs, then sentencizing the abstracts to create one example for each abstract sentence.
* For STS datasets, transformation was done by inserting random sentences from the datasets to one of the sentences in the pair to simulate a long "context". The original labels were mapped to our OOC definition. If the original pair was indicated as dissimilar, we consider the pair as OOC.

To enable training of BERT-based classifiers, each training example was split into sub-sequences of maximum 512 tokens. The OOC label for each sub-sequence was derived through a generative labelling process with Llama-3.1-70b-Instruct.

### Model Training
RAGulator is fine-tuned from `microsoft/deberta-v3-large` ([He et al., 2023](https://arxiv.org/pdf/2111.09543.pdf)).

### Model Performance
<p align="center">
    <img src="./model-performance.png" width="700">
</p>

We compare our models to LLM-as-a-judge (Llama-3.1-70b-Instruct) as a baseline. We evaluate on both a held-out data split of our simulated RAG dataset, as well as an out-of-distribution collection of private enterprise data, which consists of RAG responses from a real use case.

The deberta-v3-large variant is our best-performing model, showing a 19% increase in AUROC and a 17% increase in F1 score despite being significantly smaller than Llama-3.1.

## Installation
The RAGulator model was trained using PyTorch 1.13.1. Although the model will run on PyTorch 2.x, we strongly recommend keeping to the same version used in training.
Run the following commands to install the package in a virtualenv:
```bash
python -m venv "ragulator-env"
source ragulator-env/bin/activate
pip install "ragulator @ git+https://github.com/ipoeyke/RAGulator.git@main"
```

## Usage - batch and long-context inference
We provide a simple wrapper to demonstrate batch inference and accommodation for long-context examples.
```python
from ragulator import RAGulator

model = RAGulator(
    model_name='deberta-v3-large', # only value supported for now
    batch_size=32, # inference batch size
    device='cpu' # corresponds to torch.device
)

# input
sentences = ["This is the first sentence", "This is the second sentence"]
contexts = ["This is the first context", "This is the second context"]

# batch inference
model.infer_batch(
    sentences,
    contexts,
    return_probas=True # True for OOC probabilities, False for binary labels
)
```

## Citation
```
@misc{poey2024ragulatorlightweightoutofcontextdetectors,
      title={RAGulator: Lightweight Out-of-Context Detectors for Grounded Text Generation}, 
      author={Ian Poey and Jiajun Liu and Qishuai Zhong and Adrien Chenailler},
      year={2024},
      eprint={2411.03920},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.03920}, 
}
```