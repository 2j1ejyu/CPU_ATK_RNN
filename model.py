import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(6,dim)
        self.fc = nn.Linear(dim,1)
        

    def forward(self, x, hidden):
        outputs, hidden = self.rnn(x, hidden)
#         print(outputs)
        outputs = F.relu(outputs)
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        x = self.fc(outputs)
#         print(x.shape)
        return x, hidden
