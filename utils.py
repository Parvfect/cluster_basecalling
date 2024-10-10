
import datetime
import os
import numpy as np

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

    target_sequence = target_sequence.split()
    decoded_sequence = decoded_sequence.split()

    target_counts = np.zeros(n_motifs)
    decoded_counts = np.zeros(n_motifs)

    for i in target_sequence:
        target_counts[int(i)-1] += 1
    for i in decoded_sequence:
        decoded_counts[int(i)-1] += 1

    # we care about the counts of the motifs in the target sequence
    return sum([j if j<=i else i for i,j in zip(target_counts, decoded_counts)]) / sum(target_counts)

