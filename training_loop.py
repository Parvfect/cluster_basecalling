
import torch
import torch.nn as nn
import torch.optim as optim
from crnn import CNN_BiGRU_Classifier
import math
from tqdm import tqdm
import numpy as np
from training_data import data_preproc
from sklearn.model_selection import train_test_split
from greedy_decoder import GreedyCTCDecoder
from utils import get_actual_transcript, get_savepaths
import torchaudio
import datetime
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Running on {device}")

labels_int = np.arange(11).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

model_save_path, file_write_path = get_savepaths()
model_save_iterations = 20

# Model Parameters
input_size = 1  # Number of input channels
hidden_size = 128
num_layers = 3
output_size = 11  # Number of output classes
dropout_rate = 0.2
saved_model = False

# Model Definition
model = CNN_BiGRU_Classifier(input_size, hidden_size, num_layers, output_size, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss()

# Loading model
if saved_model:
    data_path = os.path.join("20240701.205513.995817", "model.pth")
    model_path = os.path.join(os.environ['HOME'], os.path.join("training_logs", data_path))

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




X, y = data_preproc()

# Creating Train, Test, Validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 


n_classes = 10 
step_sequence = 100
window_overlap = 50
length_per_sample = 150

epochs = 20
model_output_split_size = 2

# Add over epochs
for epoch in range(epochs):

    #################### Training Loop #################
    print(f"Epoch {epoch}")
    model = model.to(device)
    model.train()

    for i in tqdm(range(len(X_train))):

        training_sequence, target_sequence = X_train[i].to(device), torch.tensor(y_train[i]).to(device)
        input_lengths = torch.tensor(X_train[i].shape[0])
        target_lengths = torch.tensor(len(target_sequence))

        # Zero out the gradients
        optimizer.zero_grad()

        try:        
            
            # Give the model the input in chunks based on the model_output_split_size flag
            model_output_timestep = torch.zeros([input_lengths, output_size]).to(device)
            stepper_size = (input_lengths + model_output_split_size - 1) // model_output_split_size  # Adjust stepper_size to cover all elements

            for j in range(0, input_lengths, stepper_size):
                end_index = min(j + stepper_size, input_lengths)
                
                # Get the model output for the current chunk
                model_output_chunk = model(training_sequence[j:end_index])
                
                # Ensure the sizes match before assignment
                model_output_timestep[j:end_index] = model_output_chunk
        
            loss = ctc_loss(model_output_timestep, target_sequence, input_lengths, target_lengths)
            
            loss.backward()
            # Update the weights
            optimizer.step()

        except Exception as e:
            print(f"Exception {e} = {len(stepper_size)}")
            print(e)
            with open(file_write_path, 'a') as f:
                f.write(f"\nException ={e}")
                f.write(f"\nModel Output split size = {model_output_split_size}, {stepper_size}")
            model_output_split_size+=1
            continue
            
        

        if i % 100 == 0:
            with open(file_write_path, 'a') as f:
                f.write(f"\nEpoch {epoch} Batch {i} Loss {loss.item()} ")
            
        # Saving model weights
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
    correct = 0
    total = 0
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
            
            if greedy_transcript == actual_transcript:
                correct += 1

            total += 1
            val_loss += loss.item()

    val_loss /= len(X_val)
    val_accuracy = correct / total
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    with open(file_write_path, 'a') as f:
        f.write(f"\n\nEpoch {epoch} Valiation Loss {val_loss:.4f} \n")




        
        
