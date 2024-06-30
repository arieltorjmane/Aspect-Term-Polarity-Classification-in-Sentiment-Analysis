import pandas as pd
import numpy as np
import re
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tqdm import tqdm
from typing import List

from utils import *

# Define a custom dataset for loading the training samples in batches

class ABSADataset(Dataset):
    def __init__(self, df,tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Extract start and end positions from the "Position" column
        start, end = map(int, row['Position'].split(':'))

        # Insert [TGT] and [/TGT] tokens around the aspect based on the positions
        modified_text = row['Text'][:start] + '[TGT]' + row['Text'][start:end] + '[/TGT]' + row['Text'][end:]

        # Tokenize the modified text
        encoded_input = self.tokenizer(modified_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        # Extract and encode the sentiment label
        label_dict = {"negative": 0, "neutral": 1, "positive": 2}
        label = label_dict[row['Sentiment']]

        # Prepare the final output
        input_ids = encoded_input['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoded_input['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }



# Hyper-parameters
EPOCHS = 3
BATCH = 8
LR = 5e-5
WEIGHT_DECAY = 0.01


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below

    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.

        """
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)



    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_data = pd.read_csv(train_filename, delimiter='\t', header=None, names=['Sentiment', 'Category', 'Keyword', 'Position', 'Text'])

        # Initialize the model
        self.model.to(device)

        # Clean the data
        preprocess_data(train_data)

        # Create dataset
        train_dataset = ABSADataset(train_data,self.tokenizer)

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR,weight_decay = WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            self.model.train()
            train_preds, train_labels = [], []

            for batch in tqdm(train_loader):

                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # Move logits and labels to CPU for metric calculation
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                train_preds.extend(np.argmax(logits, axis=1).flatten())
                train_labels.extend(label_ids.flatten())
        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro')

        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        data = pd.read_csv(data_filename, delimiter='\t', header=None, names=['Sentiment', 'Category', 'Keyword', 'Position', 'Text'])
        preprocess_data(data)

        dataset = ABSADataset(data,self.tokenizer)
        loader = DataLoader(dataset, batch_size=BATCH)

        self.model.to(device)
        self.model.eval()

        preds = []

        for batch in tqdm(loader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                preds.extend(np.argmax(logits, axis=1).flatten())

        # Extract and encode the sentiment label
        preds = list(preds)
        label_dict = {0: "negative", 1: "neutral", 2: "positive"}
        preds = [label_dict[pred] for pred in preds]

        return preds
