# Sentiment Analysis using DistilBERT

## Overview
This is an end-to-end NLP application for sentiment analysis using a pretrained
DistilBERT transformer and a Gradio web interface.

## Model
- DistilBERT fine-tuned on SST-2
- Loaded using Hugging Face Transformers

## Why Pretrained?
Using pretrained weights avoids:
- High computational cost
- Need for large labeled datasets
- Overfitting risks

This makes it ideal for fast and reliable inference.

## Features
- Proper tokenization with attention masks
- Transformer-based inference pipeline
- Confidence thresholding ("UNCERTAIN")
- Interactive Gradio UI

## Installation and running

pip install -r requirements.txt
run app.py