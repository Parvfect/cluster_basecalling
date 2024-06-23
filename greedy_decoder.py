

import torch
import torch.nn as nn


class GreedyCTCDecoder(nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels=labels
        self.blank = blank

    def forward(self, emission:torch.Tensor):
        """Given a sequence emission over labels, get the best path"""

        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()