import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# NumPy implementation
class NumpyTransformer:
    def __init__(self, q, k, v, o):
        self.W_q = q
        self.W_k = k
        self.W_v = v
        self.W_o = o

    def softmax(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def attention(self, Q, K, V):
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        weights = self.softmax(scores)
        return np.dot(weights, V)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        attended = self.attention(Q, K, V)
        output = np.dot(attended, self.W_o)
        return output #self.sigmoid(output)

# PyTorch implementation
class PyTorchTransformer(nn.Module):
    def __init__(self, q, k, v, o):
        super(PyTorchTransformer, self).__init__()
        self.W_q = nn.Linear(q.shape[0], q.shape[1], bias=False)
        self.W_k = nn.Linear(k.shape[0], k.shape[1], bias=False)
        self.W_v = nn.Linear(v.shape[0], v.shape[1], bias=False)
        self.W_o = nn.Linear(o.shape[0], o.shape[1], bias=False)

        # Manually set the weights to match the NumPy weights
        self.W_q.weight = nn.Parameter(torch.tensor(q.T, dtype=torch.float32))
        self.W_k.weight = nn.Parameter(torch.tensor(k.T, dtype=torch.float32))
        self.W_v.weight = nn.Parameter(torch.tensor(v.T, dtype=torch.float32))
        self.W_o.weight = nn.Parameter(torch.tensor(o.T, dtype=torch.float32))

    def attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attended = self.attention(Q, K, V)
        output = self.W_o(attended)
        return output# torch.sigmoid(output)

# Initialize parameters
batch_size = 5
intput_dim = 10
hidden_dim = 20

# initialize weights
q = np.random.rand(intput_dim, hidden_dim)
k = np.random.rand(intput_dim, hidden_dim)
v = np.random.rand(intput_dim, hidden_dim)
o = np.random.rand(hidden_dim, intput_dim)

# Initialize models
numpy_model = NumpyTransformer(q, k, v, o)
torch_model = PyTorchTransformer(q, k, v, o)

# Define input data
input_data = np.random.rand(batch_size, intput_dim)
input_data_torch = torch.tensor(input_data, dtype=torch.float32)

# Forward pass
numpy_output = numpy_model.forward(input_data)
torch_output = torch_model.forward(input_data_torch).detach().numpy()

# Compare outputs
print("NumPy Output:\n", numpy_output)
print("PyTorch Output:\n", torch_output)
print("Equal?:\n", np.isclose(numpy_output, torch_output).all())
if not np.isclose(numpy_output, torch_output).all():
    print("Difference:\n", np.abs(numpy_output - torch_output))

