
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
        dataset = dataset.sample(frac=0.005, random_state=1)
    
    X = dataset[column_x].to_numpy().tolist()
    y = dataset[column_y].to_numpy()

    if payload:
        payload = dataset['Payload_Sequence'].to_numpy()
        return X, y, payload
    
    return X, y
       

def data_preproc(X, window_size=150, step_size=100):
    """
    Splits each long read of the dataset into n windows determined by the window and step size. 
    """

    # So we split and norm it
    sequences_dataset = []
    for i in tqdm(X):
        
        j = normalize([i]).flatten() # Gotta get rid of this
        sequence_length = len(j)
        n_samples = math.floor(sequence_length / step_size) # Since we don't send the last one if it is beyond the total size
        ptr, counter = 0, 0
        sequences = torch.zeros([n_samples, 1, window_size])

        # Chop the sequences into windows of length window size and with an overlap of window_size - step_size
        while ptr <= sequence_length:
            try:
                if ptr + window_size > sequence_length:
                    break  # Don't pad
                else:
                    sequence_chop = j[ptr: ptr + window_size]
            
                sequence_chop = torch.tensor(sequence_chop, dtype=torch.float32).view(1, len(sequence_chop))
                sequences[counter] = sequence_chop
            
            except IndexError:
                continue
                
            ptr += step_size
            counter+=1
        
        sequences_dataset.append(sequences)
        
    return sequences_dataset