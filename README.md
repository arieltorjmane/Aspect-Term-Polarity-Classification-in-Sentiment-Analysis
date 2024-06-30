# Aspect-Term-Polarity-Classification-in-Sentiment-Analysis

This README document provides a detailed description of the classifier developed in the scope of the CentraleSupelec NLP course assignment. The classifier implemented is designed to identify sentiments (positive, neutral, negative) expressed by specific words in the context of a given aspect pertaining to the dataset. This type of sentiment analysis task is commonly referred to as Aspect Based Sentiment Analysis (ABSA).  Below are the details of the classification model, input and feature representation, and the resources utilized. The data consists of restaurant reviews.

---

Model Overview

Model Selection: The classifier we used is a fine-tuned version of the pre-trained  hugging face transformers : "cardiffnlp/twitter-roberta-base-sentiment", a RoBERTa-based model fine-tuned for sentiment analysis on Twitter data. RoBERTa itself is an optimized version of BERT, designed for improved performance across a variety of NLP tasks.
Classification Task: the model is adapted for ABSA tasks where target tokens can either express negative, neutral or positive sentiment in respect to a particular aspect.
Environment: the Hugging Face transformers library was used, the model is compatible with PyTorch for training and inference.

---

Input and Feature Representation

Preprocessing: We include a utils.py file containing a series of data preprocessing functions. It cleans the review text by performing several operations. It converts the text to lowercase, adds space around certain punctuation marks (excluding apostrophes and punctuation marks that could express sentiment like ! and ?), removes HTML tags, keeps only letters, numbers, spaces, and replaces multiple spaces with a single space. Following these modifications, we adjust the position start and end of the aspect term to reflect its new position. We obtain a final training set including polarity, aspect term, position, and text. 

Aspect Highlighting: The input text is pre-processed to include [TGT] and [/TGT] tokens surrounding the target aspect term. This approach allows the model to identify the target tokens to classify.
Tokenization: Utilizing the tokenizer from the pre-trained model, texts are converted into sequences of token IDs, with attention masks generated to distinguish real data from padding in fixed-length sequences.

Feature Extraction: Through self-attention mechanisms, the model inherently extracts contextual features from the input, capturing intricate relationships and sentiments related to the highlighted aspect.

---

Resources

transformers Library: Essential for accessing pre-trained models and their tokenizers, the Hugging Face transformers library is a core component of our classifier, facilitating model loading, tokenization, and fine-tuning processes.

ABSADataset Class: A custom PyTorch Dataset implementation for handling data loading, preprocessing, and preparation for training and prediction phases, aligning with the specific needs of ABSA.

---

Training and Evaluation:

DataLoader: Facilitates efficient batch processing of training data.
Optimizer: AdamW optimizer is used with defined learning rates and weight decay parameters for regularization.

Metrics: Training progress is monitored through accuracy, precision, recall, and F1-score, ensuring comprehensive evaluation of model performance.

---

Model Accuracy

After training our ABSA classifier, we evaluated its performance on the dev, which served as our testing ground. The model achieved an accuracy of 86.44%.

---

Summary

This ABSA classifier harnesses the power of transformer-based models for fine-grained sentiment analysis. By focusing on aspect-specific sentiments within texts, it opens up possibilities for deep sentiment analysis in various applications, from customer feedback analysis to opinion mining in social media content.
