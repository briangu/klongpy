import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        # self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Reshape input_seq to (seq_len, batch, input_size)
        input_seq = input_seq.view(len(input_seq), 1, -1)
        lstm_out, _ = self.lstm(input_seq)
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        predictions = self.linear(lstm_out[-1].view(1, -1))
        return predictions[:, 0]


class LSTMWrapper:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.model = LSTMModel(input_size, hidden_layer_size, output_size)


class Trainer:

    def __init__(self, wrapper):
        self.model = wrapper.model
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def __call__(self, x, y):
        x = torch.tensor(x).float()
        y = torch.tensor([y]).float()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        single_loss = self.loss_function(y_pred, y)
        single_loss.backward()
        self.optimizer.step()
        return single_loss.item()


def load(filename):
    model =torch.load(filename)
    model.eval()
    return model

def save(filename, model):
    torch.save(model, filename)


klongpy_exports = {"load": load, "model": LSTMWrapper, "trainer": Trainer}
