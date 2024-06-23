
import torch
import torch.nn as nn


class CNN_BiGRU_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):

        super(CNN_BiGRU_Classifier, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected Layer
        self.fc = nn.Linear(64, hidden_size)

        # Bidirectional GRU layers
        self.bigru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Sequence Classifier
        self.output = nn.Linear(hidden_size*2, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        
        # CNN layers
        x = self.conv1(x)
        x = torch.relu(x)
        #x = self.norm1(x)
        x = self.pool(x)

        # Apply LayerNorm after permuting the dimensions
        #x = x.permute(0, 2, 1)
        #x = self.norm1(x)

        x = self.conv2(x)
        x = torch.relu(x)

        # Fully connected layer
        x = x.permute(0, 2, 1)
        x = self.fc(x)

        # Bidirectional GRU layers
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        x, _ = self.bigru(x, h0)

        # Output layer
        x = self.output(x)

        x = x.sum(dim=1)
        return x


if __name__ == "__main__":     
    # Example usage
    input_size = 1  # Number of input channels
    hidden_size = 128
    num_layers = 3
    output_size = 11  # Number of output classes
    dropout_rate = 0.2
    
    batch_size = 200

    model = CNN_BiGRU_Classifier(input_size, hidden_size, num_layers, output_size, dropout_rate)
    # Define the input tensor
    sequence_length = 150 # Chop up window is going to be the same length
    input_tensor = torch.randn(batch_size, input_size, sequence_length)  # (sequence_length)
    # Reshape the input array into a 3D tensor
    output = model(input_tensor)
    print(output.shape)
    print(output)