
import datetime
import os

def get_model_savepath():
    savepath = r"C:\Users\Parv\Doc\HelixWorks\Basecalling\code\pytorch_examples\model_runs"
    
    uid = str(datetime.datetime.now()).replace(' ', '.').replace('-','').replace(':',"")
    savepath = os.path.join(savepath, f"{uid}")
    os.mkdir(savepath)

    return os.path.join(savepath, "model.pth")

def get_actual_transcript(target_sequence):
    """Gets the tensor target sequence and returns the transcript"""

    target_list = target_sequence.tolist()
    seq = ""

    for i in target_list:
        seq += f" {i}"

    return seq
