
import datetime
import os
import numpy as np
import torch
import copy

def get_savepaths():

    uid = str(datetime.datetime.now()).replace(' ', '.').replace('-','').replace(':',"")

    savepath = os.path.join(os.environ['HOME'], os.path.join("training_logs", f"{uid}"))
    
    os.mkdir(savepath)

    model_savepath = os.path.join(savepath, "model.pth")
    file_savepath = os.path.join(savepath, "log.txt")

    return model_savepath, file_savepath

def get_actual_transcript(target_sequence):
    """Gets the tensor target sequence and returns the transcript"""

    target_list = target_sequence.tolist()
    seq = ""

    for i in target_list:
        seq += f" {i}"

    return seq


def get_motifs_identified(target_sequence, decoded_sequence, n_motifs=19):
    """ Got to do it by payload"""

    target_cycles = sort_transcript(target_sequence)
    payload_cycles = sort_transcript(decoded_sequence)

    n_motifs = 0
    found_motifs = 0
    motif_errs = 0
    payload_acc = 0
    payload_err = 0
    n_cycles = 0


    for i in range(len(target_cycles)):
        found_motifs_arr = [motif for motif in payload_cycles[i] if motif in target_cycles[i]]
        motif_errors = [motif for motif in payload_cycles[i] if motif not in target_cycles[i]]

        n_motifs += len(target_cycles[i])
        found_motifs += len(found_motifs_arr)
        motif_errs += len(motif_errors)

        if len(target_cycles[i]) > 0:
            n_cycles += 1
            payload_acc += (len(found_motifs_arr)) / len(target_cycles[i])
            payload_err += len(motif_errors) / len(target_cycles[i])

    # Let's do motif errors per found motif
    if n_motifs > 0 and (found_motifs + motif_errs) > 0:
        motif_acc_cycle = found_motifs/n_motifs
        motif_err_cycle = motif_errs/(found_motifs + motif_errs)
    else:
        motif_acc_cycle = 0.0
        motif_err_cycle = 0.0

    if n_cycles > 0:
        motif_acc_payload = payload_acc/n_cycles
        motif_err_payload = payload_err/n_cycles
    else:
        motif_acc_payload = 0.0
        motif_err_payload = 0.0

    return (motif_acc_cycle, motif_err_cycle, motif_acc_payload, motif_err_payload)


def sort_transcript(transcript):

    cycles = [[] for i in range(8)]
    
    if type(transcript) == str:
        transcript = transcript.split()
    
    split_transcript = [int(i) for i in transcript if i != '']
    
    for i in range(len(split_transcript)):

        found_motif = split_transcript[i]

        # If we have a payload motif
        if found_motif < 9:

            # finding the spacers - only for payload cycles
            if i > 0:

                # Checking for Back Spacer
                if split_transcript[i-1] > 10:
                    cycle_number = split_transcript[i-1] - 11
                    cycles[cycle_number].append(split_transcript[i])

                # Checking for Forward Spacer
                elif i < len(split_transcript) - 1:
                    if split_transcript[i+1] > 10:
                        cycle_number = split_transcript[i+1] - 11
                        cycles[cycle_number].append(split_transcript[i])

            else:
                if i < len(split_transcript) - 1:
                    # Checking for Forward Spacer
                    if split_transcript[i+1] > 10:
                        cycle_number = split_transcript[i+1] - 11
                        cycles[cycle_number].append(split_transcript[i])   

    return cycles


def create_spacer_sequence(cycles):

    spacer_sequence = []

    cycle_number = 11

    """
    if cycles.isinstance(str):
        payload = eval(cycles)
    """
        
    for i in cycles:
        for j in i:
            spacer_sequence.append(cycle_number)
            spacer_sequence.append(j)
            spacer_sequence.append(cycle_number)
        cycle_number += 1

    return spacer_sequence


def gt_loss(
        ctc_loss, model_output_timestep, target_sequence, payload_sequence,
        device, input_lengths):

    target_cycles = sort_transcript(target_sequence)
    payload_cycles = sort_transcript(payload_sequence)

    additional_cycles = []

    for i in range(len(target_cycles)):
        unique_motifs = [motif for motif in payload_cycles[i] if motif not in target_cycles[i]]
        for k in unique_motifs:
            new_cycle = copy.deepcopy(target_cycles)
            new_cycle[i].append(k)
            additional_cycles.append(new_cycle)

    additional_cycles = [
        torch.tensor(
            create_spacer_sequence(i)).to(device) for i in additional_cycles]

    return [
        ctc_loss(
            model_output_timestep, i, input_lengths, torch.tensor(
                len(i))) for i in additional_cycles]



def display_metrics(
        file_write_path, greedy_transcript, actual_transcript, target_metrics,
        loss, payload_transcript=None, payload_metrics=None, type=0, epoch=None, batch=None):

    with open(file_write_path, 'a') as f:

        if type == 0:  # Train
            f.write(
                f"\nEpoch {epoch} \nBatch {batch} \nLoss {loss}\n")
            print(
                f"\nEpoch {epoch} \nBatch {batch} \nLoss {loss}\n")
            
        elif type == 1:  # Validation
            f.write(
                f"\nValidation Epoch {epoch} \nLoss {loss}\n")
            print(
                f"\nValidation Epoch {epoch} \nLoss {loss}\n")
        elif type == 2:  # Test
            f.write(
                f"\nTest Loss {loss}\n")
            print(
                f"\nTest Loss {loss}\n")
            
        else:  # Unseen Data
            f.write(
                f"\nUnseen Data Loss {loss}\n")
            print(
                f"\nUnseen Data Loss {loss}\n")

        f.write(f"Payload Transcript - {payload_transcript}\n")
        f.write(f"Actual Transcript -{actual_transcript}\n")
        f.write(f"Greedy Transcript  - {greedy_transcript}\n")
        f.write(f"Motif Acc/Err Cycle/Payload Target Seq - {target_metrics}\n")
        f.write(f"Motif Acc/Err Cycle/Payload Payload Seq - {payload_metrics}\n")

    print(f"Payload - {payload_transcript}")
    print(f"Actual Transcript -{actual_transcript}")
    print(f"Greedy Transcript  - {greedy_transcript}")
    print(f"Motif Acc/Err Cycle/Payload Target Seq - {target_metrics}")
    print(f"Motif Acc/Err Cycle/Payload Payload Seq - {payload_metrics}")


def get_metrics_for_evaluation(
        greedy_decoder, model_output_timestep, target_sequence, loss, payload_sequence=None):

    greedy_result = greedy_decoder(model_output_timestep)
    greedy_transcript = " ".join(greedy_result)
    actual_transcript = get_actual_transcript(target_sequence)
    target_metrics = get_motifs_identified(
        actual_transcript, greedy_transcript)

    if payload_sequence:
        payload_transcript = get_actual_transcript(payload_sequence)
        payload_metrics = get_motifs_identified(
            payload_transcript, greedy_transcript)
        return greedy_transcript, actual_transcript, payload_transcript, target_metrics, payload_metrics
    
    return greedy_transcript, actual_transcript, target_metrics


def get_bases_identified(target_seq, decoded_seq):
    return sum([i==j for i, j in zip(target_seq, decoded_seq)])/len(target_seq)