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

## Installation

1. Clone this repository:
