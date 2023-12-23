import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def train(model, input_seq, label_seq, epochs):
        epoch_losses = []
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for i in range(epochs):
            optimizer.zero_grad()
            y_pred = model(input_seq)
            single_loss = loss_function(y_pred, label_seq)
            single_loss.backward()
            optimizer.step()
            if i % (max(1,epochs//10)) == 0:
                epoch_losses.append(single_loss.item())
                # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        return epoch_losses

    def save(self, filename):
        torch.save(self, filename)

    def predict(self, input_seq):
        return self(input_seq)


def create_model(input_size, hidden_layer_size, output_size):
    return LSTMModel(input_size, hidden_layer_size, output_size)


def load(filename):
    model =torch.load(filename)
    model.eval()
    return model


klongpy_exports = {"load": load, "create": create_model}
