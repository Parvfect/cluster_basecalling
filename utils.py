
import datetime
import os

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
