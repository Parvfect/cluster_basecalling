
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from crnn import CNN_BiGRU_Classifier
import math
from tqdm import tqdm
import numpy as np
from training_data import data_preproc, load_training_data
from sklearn.model_selection import train_test_split
from greedy_decoder import GreedyCTCDecoder
from utils import get_actual_transcript, get_savepaths, get_motifs_identified, \
    gt_loss, get_bases_identified
import torchaudio
import datetime
import os
from utils import get_savepaths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Running on {device}")

model_save_path, file_write_path = get_savepaths()

model_save_iterations = 2000

display_iterations = 5000


output_classes = 5 # including blank
# Model Parameters
input_size = 1  # Number of input channels
hidden_size = 256
num_layers = 4
output_size = output_classes  # Number of output classes
dropout_rate = 0.2
saved_model = False
save_model = True
alpha = 0
n_classes = output_classes
step_sequence = 5
length_per_sample = 20

"""
#Motif level - comment out as needed
step_sequence = 100
window_overlap = 50
length_per_sample = 500
output_classes = 14 # including blank
output_size = output_classes  # Number of output classes
"""

labels_int = np.arange(output_classes).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

# Model Definition
model = CNN_BiGRU_Classifier(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

epochs = 30
model_output_split_size = 1

dataset_path = os.path.join(os.environ['HOME'], "chopped_synthetic_base_level.pkl")


# Currently at motif level
X, y = load_training_data(dataset_path=dataset_path, sample=False)


X = data_preproc(X, window_size=length_per_sample, step_size=step_sequence)

# Creating Train, Test, Validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
torch.autograd.set_detect_anomaly(True)

training_flag = True


if training_flag:
    # Add over epochs
    for epoch in range(epochs):

        #################### Training Loop #################
        print(f"Epoch {epoch}")
        model.train()

        for i in tqdm(range(len(X_train))):

            try:
                training_sequence = X_train[i].to(device)
                target_sequence = torch.tensor(y_train[i]).to(device)

                
                input_lengths = torch.tensor(X_train[i].shape[0])
                target_sequence = torch.tensor(y_train[i]).to(device)

                input_lengths = torch.tensor(X_train[i].shape[0])
                target_lengths = torch.tensor(len(target_sequence))

                # Zero out the gradients
                optimizer.zero_grad()

                model_output_timestep = model(training_sequence)
                loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)

                # Training on ground truth
                loss.backward()
                # Update the weights
                optimizer.step()

            except Exception as e:
                print(e)
                with open(file_write_path, 'a') as f:
                    f.write(f"\nException ={e}")
                    print(target_sequence)
                    continue
               

            if i % display_iterations == 0:
                greedy_result = greedy_decoder(model_output_timestep)
                greedy_transcript = " ".join(greedy_result)
                actual_transcript = get_actual_transcript(target_sequence)
                #actual_transcript = get_actual_transcript(payload_sequence)
                motif_err = torchaudio.functional.edit_distance(actual_transcript, greedy_transcript) / len(actual_transcript)
                motifs_identifed = get_bases_identified(actual_transcript, greedy_transcript)

                print(f"Motifs identified {motifs_identifed}")
                print(motif_err)
                print(loss)


                with open(file_write_path, 'a') as f:
                    f.write(f"\nEpoch {epoch} Batch {i} Main Loss {loss.item()} \n")
                    f.write(f"Transcript: {greedy_transcript}\n")
                    f.write(f"Actual Transcript: {actual_transcript}\n")
                    f.write(f"Sequence edit distance: {motif_err}\n")
                    f.write(f"Motifs Identified: {motifs_identifed}\n")
                

                
            # Saving model weights
            if save_model:
                if i % model_save_iterations == 0:
                    torch.save({
                    'epoch': epoch,
                    'batch':i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, model_save_path)

        
        ################## Validation Loop ####################
        
        model.eval()
        val_loss = 0.0
        val_acc = []
        motifs_identified_arr = []
        with torch.no_grad():
            for i in tqdm(range(len(X_val))):

                validation_sequence, target_sequence = torch.tensor(X_val[i]).to(device), torch.tensor(y_val[i]).to(device)
                
                try:
                    model_output_timestep = model(validation_sequence) # Getting model output
                except:
                    continue

                input_lengths = torch.tensor(X_val[i].shape[0])
                target_lengths = torch.tensor(len(target_sequence))
                
                loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)

                greedy_result = greedy_decoder(model_output_timestep)
                greedy_transcript = " ".join(greedy_result)
                actual_transcript = get_actual_transcript(target_sequence)
                #actual_transcript = get_actual_transcript(payload_sequence)
                motif_err = torchaudio.functional.edit_distance(actual_transcript, greedy_transcript) / len(actual_transcript)
                motifs_identified = get_bases_identified(actual_transcript, greedy_transcript)
                val_acc.append(motif_err)
                motifs_identified_arr.append(motifs_identified)
                
                val_loss += loss.item()

        val_loss /= len(X_val)
        val_accuracy = np.mean(val_acc)
        motifs_identified = np.mean(motifs_identified_arr)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Edit Distance: {val_accuracy:.4f}")
        print(f"Motifs Identified: {motifs_identified:.4f}")

        with open(file_write_path, 'a') as f:
            f.write(f"\n\nEpoch {epoch} Validation Loss {val_loss:.4f}, Validation Edit Distance: {val_accuracy:.4f} \n Motifs Identified {motifs_identified:.4f}")
        

# Test Loop
model.eval()
test_loss = 0.0
distances_arr = []
motifs_identifed_arr = []
with torch.no_grad():
    for i in tqdm(range(len(X_test))):

        test_sequence, target_sequence = torch.tensor(X_test[i]).to(device), torch.tensor(y_test[i]).to(device)

        try:
            model_output_timestep = model(test_sequence) # Getting model output
        except:
            continue

        input_lengths = torch.tensor(X_test[i].shape[0])
        target_lengths = torch.tensor(len(target_sequence))

        loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)
        #loss = ctc_loss(model_output_timestep, payload_sequence, input_lengths, payload_lengths)
        test_loss += loss.item()

        greedy_result = greedy_decoder(model_output_timestep)
        greedy_transcript = " ".join(greedy_result)
        actual_transcript = get_actual_transcript(target_sequence)
        #actual_transcript = get_actual_transcript(payload_sequence)
        
        motif_err = torchaudio.functional.edit_distance(actual_transcript, greedy_transcript) / len(actual_transcript)
        distances_arr.append(motif_err)

        motifs_identifed = get_bases_identified(actual_transcript, greedy_transcript)
        motifs_identifed_arr.append(motifs_identifed)

test_loss /= len(X_test)
test_accuracy = np.mean(distances_arr)
motifs_identifed = np.mean(motifs_identifed_arr)

with open(file_write_path, 'a') as f:
    f.write(f"Test Loss: {test_loss:.4f}, Test Edit Distance: {test_accuracy:.4f}, Motifs Identified: {motifs_identifed:.4f}")




        
        
