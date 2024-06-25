
from utils import get_savepaths
import torch

model_savepath, file_savepath = get_savepaths()

with open(file_savepath, 'w') as f:
    f.write(torch.cuda.is_available())
    f.write(torch.cuda.get_device_name(0))
    f.write()

    


