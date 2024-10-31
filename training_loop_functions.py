


import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from crnn import CNN_BiGRU_Classifier
import math
from tqdm import tqdm
import numpy as np
from training_data import load_training_data, data_preproc
from sklearn.model_selection import train_test_split
from greedy_decoder import GreedyCTCDecoder
from utils import get_actual_transcript, get_savepaths, get_motifs_identified, \
    gt_loss, get_metrics_for_evaluation, display_metrics, create_spacer_sequence
import datetime
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Running on {device}")

output_classes = 19 # including blank

labels_int = np.arange(output_classes).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

#model_save_path, file_write_path = get_savepaths()
file_write_path = "training_logs.txt"

model_save_iterations = 5000
display_iterations = 10000

# Model Parameters
input_size = 1  # Number of input channels
hidden_size = 256
num_layers = 4
output_size = output_classes  # Number of output classes
dropout_rate = 0.2
saved_model = True
save_model = True
alpha = 0.02
model_path = "model_underfit.pth"
#test_data_path = 'sampled_test_dataset_v4_spacers.pkl'
n_classes = output_classes
step_sequence = 100
window_overlap = 50
length_per_sample = 150
ctc_loss = nn.CTCLoss(
    blank=0, reduction='mean', zero_infinity=True)

torch.autograd.set_detect_anomaly(True)


def set_variables(msp, tdp, mp, sm, fwp):

    global model_save_path, test_data_path, model_path, saved_model, file_write_path
    model_save_path = msp
    test_data_path = tdp
    model_path = mp
    saved_model = sm
    file_write_path = fwp
    

def prepare_data_for_training(dataset_path, sample):

    # Model Definition
    model = CNN_BiGRU_Classifier(
        input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Loading model
    if saved_model:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    X, y, payload_label = load_training_data(dataset_path, sample=sample)
    X, y, payload_label = data_preproc(X, y, payload_label, chop_reads=1)

    # Creating Train, Test, Validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=1) 

    X_train, X_test, payload_train, payload_test = train_test_split(X, payload_label, test_size=0.2, random_state=42)
    X_train, X_val, payload_train, payload_val = train_test_split(X_train, payload_train, test_size=0.10, random_state=1)

    unseen_data = pd.read_pickle(test_data_path)
    unseen_data = unseen_data.sample(frac=0.01, random_state=1)
    test_X = unseen_data['squiggle'].to_numpy()
    test_y = unseen_data['Spacer_Sequence'].to_numpy()
    test_payload = unseen_data['Payload'].to_numpy()

    test_X, test_y, test_payload = data_preproc(test_X, test_y, test_payload, chop_reads=1)

    return X_train, X_test, X_val, y_train, y_test, y_train, y_val, payload_train, payload_test, payload_val, test_X, test_y, test_payload, model, optimizer

def unseen_data_test_loop(test_X, test_y, test_payload, alpha, model):
    
    test_loss = 0.0
    target_metrics_arr = []
    payload_metrics_arr = []

    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            
            payload_spacer_sequence = create_spacer_sequence(test_payload[i])

            test_sequence = torch.tensor(test_X[i]).to(device)
            target_sequence = torch.tensor(test_y[i]).to(device)
            payload_sequence = torch.tensor(payload_spacer_sequence).to(device)

            payload_str = " ".join(str(k) for k in payload_spacer_sequence)

            model_output_timestep = model(test_sequence)  # Getting model output

            input_lengths = torch.tensor(test_X[i].shape[0])
            target_lengths = torch.tensor(len(target_sequence))

            loss = ctc_loss(
                model_output_timestep, target_sequence, input_lengths, target_lengths)
            
            if alpha > 0:
                gt_loss_arr = gt_loss(
                        ctc_loss, model_output_timestep, test_y[i],payload_str,
                        device, input_lengths)

                loss = loss + alpha*sum(gt_loss_arr)

            greedy_transcript, actual_transcript, payload_transcript, target_metrics, payload_metrics = get_metrics_for_evaluation(
                        greedy_decoder,
                        model_output_timestep, target_sequence,
                        payload_sequence, loss)
                    
            test_loss += loss.item()

            target_metrics_arr.append(target_metrics)
            payload_metrics_arr.append(payload_metrics)
            
            test_loss += loss.item()

    test_loss /= len(test_X)
    target_metrics = np.mean(np.array(target_metrics_arr), axis=0)
    payload_metrics = np.mean(np.array(target_metrics_arr), axis=0)
        
    display_metrics(
        file_write_path, greedy_transcript, actual_transcript,
        payload_transcript, target_metrics, payload_metrics, test_loss,
        type=3)


def train_model(
        X_train, X_val, y_train, y_val, payload_train, payload_val, test_X, test_y, test_payload, epochs, model,
        optimizer, alpha):

    model_output_split_size = 1
    for epoch in range(epochs):

        model.train()
        for i in tqdm(range(len(X_train))):

            training_sequence = X_train[i].to(device)
            target_sequence = torch.tensor(y_train[i]).to(device)
            payload_sequence = torch.tensor(payload_train[i]).to(device)
            input_lengths = torch.tensor(X_train[i].shape[0])
            target_lengths = torch.tensor(len(target_sequence))

            optimizer.zero_grad()

            try:
                model_output_timestep = torch.zeros([input_lengths, output_size]).to(device)
                stepper_size = (
                    input_lengths + model_output_split_size - 1) // model_output_split_size

                for j in range(0, input_lengths, stepper_size):
                    end_index = min(j + stepper_size, input_lengths)
                    model_output_chunk = model(training_sequence[j:end_index])
                    model_output_timestep[j:end_index] = model_output_chunk

                loss = ctc_loss(
                    model_output_timestep, target_sequence, input_lengths, target_lengths)
                
                if alpha > 0:
                    gt_loss_arr = gt_loss(
                        ctc_loss, model_output_timestep, y_train[i], payload_train[i],
                        device, input_lengths)
                    loss = loss + alpha*sum(gt_loss_arr)
                    
                loss.backward()
                optimizer.step()

            except Exception as e:
                model_output_split_size += 1
                with open(file_write_path, 'a') as f:
                    f.write(f"\nException ={e}")
                    f.write(f"\nModel Output split size = {model_output_split_size}, {stepper_size}")
                continue
            
            if i % display_iterations == 0:
                
                greedy_transcript, actual_transcript, payload_transcript, target_metrics, payload_metrics = get_metrics_for_evaluation(
                    greedy_decoder, model_output_timestep, target_sequence, payload_sequence, loss)
                display_metrics(
                    file_write_path, greedy_transcript, actual_transcript,
                    payload_transcript, target_metrics, payload_metrics,
                    loss, type=0, epoch=epoch, batch=i)
                
            if save_model:
                if i % model_save_iterations == 0:
                    torch.save({
                                'epoch': epoch,
                                'batch': i,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, model_save_path)

        model.eval()
        val_loss = 0.0
        target_metrics_arr = []
        payload_metrics_arr = []

        with torch.no_grad():
            for i in tqdm(range(len(X_val))):

                validation_sequence = torch.tensor(X_val[i]).to(device)
                target_sequence = torch.tensor(y_val[i]).to(device)
                payload_sequence = torch.tensor(payload_val[i]).to(device)

                model_output_timestep = model(validation_sequence)
            
                input_lengths = torch.tensor(X_val[i].shape[0])
                target_lengths = torch.tensor(len(target_sequence))
                
                loss = ctc_loss(
                    model_output_timestep, target_sequence, input_lengths, target_lengths)
                
                if alpha > 0:

                    gt_loss_arr = gt_loss(
                        ctc_loss, model_output_timestep, y_val[i], payload_val[i],
                        device, input_lengths)

                    loss = loss + alpha * sum(gt_loss_arr)

                greedy_transcript, actual_transcript, payload_transcript, target_metrics,
                payload_metrics = get_metrics_for_evaluation(
                    greedy_decoder, model_output_timestep, target_sequence,
                    payload_sequence, loss)
                
                val_loss += loss.item()

                target_metrics_arr.append(target_metrics)
                payload_metrics_arr.append(payload_metrics)
                
            val_loss /= len(X_val)
            target_metrics = np.mean(np.array(target_metrics_arr), axis=0)
            payload_metrics = np.mean(np.array(target_metrics_arr), axis=0)
            
            display_metrics(
                file_write_path, greedy_transcript, actual_transcript,
                payload_transcript, target_metrics, payload_metrics, val_loss,
                type=1, epoch=epoch)
            
            unseen_data_test_loop(test_X, test_y, test_payload, alpha, model)

    return model
        

def test_model(model, X_test, y_test, payload_test, test_X, test_y, test_payload, alpha):

    model.eval()
    test_loss = 0.0
    target_metrics_arr = []
    payload_metrics_arr = []
    greedy_transcript = ''
    actual_transcript = ''
    payload_transcript = ''

    with torch.no_grad():
        for i in tqdm(range(len(X_test))):

            test_sequence = torch.tensor(X_test[i]).to(device)
            target_sequence = torch.tensor(y_test[i]).to(device)
            payload_sequence = torch.tensor(payload_test[i]).to(device)

            model_output_timestep = model(test_sequence)  # Getting model output

            input_lengths = torch.tensor(X_test[i].shape[0])
            target_lengths = torch.tensor(len(target_sequence))

            loss = ctc_loss(
                model_output_timestep, target_sequence, input_lengths, target_lengths)
            
            if alpha > 0:
                gt_loss_arr = gt_loss(
                        ctc_loss, model_output_timestep, y_test[i], payload_test[i],
                        device, input_lengths)

                loss = loss + alpha*sum(gt_loss_arr)

            greedy_transcript, actual_transcript, payload_transcript, target_metrics, payload_metrics = get_metrics_for_evaluation(
                        greedy_decoder, model_output_timestep, target_sequence,
                        payload_sequence, loss)
                    
            test_loss += loss.item()

            target_metrics_arr.append(target_metrics)
            payload_metrics_arr.append(payload_metrics)
            
            test_loss += loss.item()

    test_loss /= len(X_test)
    target_metrics = np.mean(np.array(target_metrics_arr), axis=0)
    payload_metrics = np.mean(np.array(target_metrics_arr), axis=0)
        
    display_metrics(
        file_write_path, payload_transcript, actual_transcript,
        greedy_transcript, target_metrics, payload_metrics, test_loss,
        type=2)
    
    unseen_data_test_loop(test_X, test_y, test_payload, alpha, model)





        
        
