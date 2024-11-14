
import torch
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import math
import os
import pickle

def load_training_data(dataset_path=None, column='Spacer_Sequence', sample=False):

    if not dataset_path:
        dataset_path = os.path.join(os.environ['HOME'], "empirical_train_dataset_v5_payload_seq.pkl")
        
    dataset = pd.read_pickle(dataset_path)

    if sample:
        dataset = dataset.sample(frac=0.005, random_state=1)
    
    X = dataset['Squiggle'].to_numpy().tolist()

    #payload = dataset['Payload_Sequence'].to_numpy()

    y = dataset[column].to_numpy()

    return X, y#, payload
       

def data_preproc(X, y, payload, chop_reads=1):
    
    n_classes = 10
    step_sequence = 100
    window_overlap = 50
    length_per_sample = 150

    if chop_reads < 1:
      y = y[:int(len(X)*chop_reads)]
      X = X[:int(len(X)*chop_reads)]
      payload = payload[:int(len(X)*chop_reads)]

    # So we split and norm it
    sequences_dataset = []
    for i in tqdm(X):
        
        j = normalize([i]).flatten()
        #j = i

        sequence_length = len(j)
            
        n_samples = math.ceil(sequence_length/step_sequence) # Since we send the last one even if it is small as can be

        ptr = 0
        counter = 0
        sequences = torch.zeros([n_samples, 1, length_per_sample])
        while ptr <= sequence_length:
            
            try:
                if ptr + length_per_sample > sequence_length:
                    break
                else:
                    sequence_chop = j[ptr:ptr+length_per_sample]
                    
                sequence_chop = torch.tensor(sequence_chop, dtype=torch.float32).view(1, len(sequence_chop))

                sequences[counter] = sequence_chop
            except IndexError:
                continue
                
            
            ptr += step_sequence
            counter+=1
        
        sequences_dataset.append(sequences)
        
         
    return sequences_dataset, y, payload

