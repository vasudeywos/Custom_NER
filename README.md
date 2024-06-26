# Custom_NER
Custom Entity extraction project, on a self-created IOB annotated datatset(using Doccano), by using Hugginface BERT transformer.

# Custom NER Model for Scientific Text Analysis

## Overview

This project implements a custom Named Entity Recognition (NER) model using BERT (Bidirectional Encoder Representations from Transformers) for identifying and classifying named entities in scientific texts. The model is specifically designed to recognize affiliations and potentially other relevant entities in academic papers.

## Model Architecture

- Base Model: BERT (bert-base-uncased)
- Task-specific layer: Token Classification Head
- Number of labels: At least 2 (Including 'O' for non-entity and 'I-AFFILIATIONS' for affiliations)

## Features

- Custom NER model trained on scientific text data
- Identification of affiliations in academic papers
- Potential for expanding to other relevant entities
- Fine-tuned BERT model for improved performance on domain-specific text

## Usage

### Training the Model

To train the model on your own dataset:

1. Prepare your data in the JSONL format (see `dataset.jsonl` for an example).
2. Run the training script:
   python train_ner_model.py
### Inference

To use the trained model for inference:

1. Load the model:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")
```
2. Perform inference on new text:
```python
  text = "Your scientific text here"
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
      logits = model(**inputs).logits
  predictions = torch.argmax(logits, dim=2)
  predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
```
## Model Performance

Training time: 211.5674 seconds
Training samples per second: 7.156
Training steps per second: 0.454
Final training loss: 0.12777168552088079
Validation Loss: Epoch 1 - 0.051182, Epoch 2 - 0.036526

## Future Work

Implement cross-validation for more robust model evaluation
Experiment with domain-specific pre-trained models (e.g., DilBERT)
Increase dataset size and diversity
Develop post-processing for entity span consolidation
Address class imbalance and introduce more granular entity types

## Acknowledgements
Hugging Face Transformers library
BERT model developers
Contributors to the scientific text dataset used for training



