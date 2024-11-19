
import torch
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import math
import os
import pickle

def load_training_data(
    dataset_path=None, column_x='Squiggle', column_y='Bases', sample=False, payload=False):

    if not dataset_path:
        dataset_path = os.path.join(os.environ['HOME'], "empirical_train_dataset_v5_payload_seq.pkl")
        
    dataset = pd.read_pickle(dataset_path)

    if sample:
        dataset = dataset.sample(frac=0.1, random_state=1)
    
    X = dataset[column_x].to_numpy().tolist()
    y = dataset[column_y].to_numpy()
    
    bases = {'A': 1, "C": 2, "T": 3, "G": 4}

    # Making sure the label is a list sequence and not  a string
    if type(y[0]) == str:
        y = [list(i) for i in y]
        y = [[bases[base] for base in i] for i in y]

    if payload:
        payload = dataset['Payload_Sequence'].to_numpy()
        return X, y, payload
    
    return X, y
       

def data_preproc(X, window_size=150, step_size=100):
    """
    Splits each long read of the dataset into n windows determined by the window and step size. 
    """

    sequences_dataset = []

    for seq in tqdm(X):
        # Normalize and flatten sequence
        #j = normalize([seq]).flatten()  # Consider vectorized normalization
        j = seq
        sequence_length = len(j)
        
        # Calculate start indices for all windows
        start_indices = range(0, sequence_length - window_size + 1, step_size)
        windows = [j[start:start + window_size] for start in start_indices]

        # Convert to a PyTorch tensor in one step
        sequences = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
        sequences_dataset.append(sequences)

    return sequences_dataset