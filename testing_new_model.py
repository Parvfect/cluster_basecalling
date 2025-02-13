

# fuck jupyter
from crnn import CNN_BiGRU_Classifier
from training_data import load_training_data, data_preproc, create_label_for_training
import pandas as pd
from greedy_decoder import GreedyCTCDecoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np


# Loading the training data
dataset_path = r"C:\Users\Parv\Doc\HelixWorks\Basecalling\code\datasets\synthetic\working_datasets\unnormalized\synth_dataset.pkl"
df = pd.read_pickle(dataset_path)
print(df.head())
X, y = load_training_data(dataset_path, column_x='Squiggle', column_y='Motifs')

n_classes = 10
step_size = 500
window_size = 5000

X = data_preproc(X, window_size, step_size, normalize_values=True)
y = create_label_for_training(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
torch.autograd.set_detect_anomaly(True)

# Model parameters
n_classes = 10
epochs = 35
model_output_split_size = 1
input_size = 1  # Number of input channels
hidden_size = 256
num_layers = 4
output_size = n_classes  # Number of output classes
dropout_rate = 0.2
n_windows = 20 # Number of adjustable windows for the model (3 timesteps)
saved_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Running on {device}")

# Model Definition
model = CNN_BiGRU_Classifier(input_size, hidden_size, num_layers, output_size, n_windows, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# CTC Loss and decoder
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
labels_int = np.arange(n_classes).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

# Add over epochs
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    for i in tqdm(range(len(X_train))):

        training_sequence, target_sequence = X_train[i].to(device), torch.tensor(y_train[i]).to(device)
        #payload_sequence = torch.tensor(payload_train[i]).to(device)
        print(training_sequence.shape)
        
        input_lengths = torch.tensor(X_train[i].shape[0])
        target_lengths = torch.tensor(len(target_sequence))

        # Zero out the gradients
        optimizer.zero_grad()

        try:        
            
            print(training_sequence.shape)
            model_output_timestep = model(training_sequence[:4])
            print(model_output_timestep.shape)

            exit()

            loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)

            # Training on ground truth
            loss.backward()
            # Update the weights
            optimizer.step()

        except Exception as e:
            print(e)
            """
            with open(file_write_path, 'a') as f:
                f.write(f"\nException ={e}")
                f.write(f"\nModel Output split size = {model_output_split_size}, {stepper_size}")
            model_output_split_size+=1
            """
            continue
            
        

