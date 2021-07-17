import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    input shape: batch_size, step, 1
    after one_hot function: batch_size, step, num_tokens
    after rnnlayer: step*batch_size, num_tokens
    output shape: step*batch_size, num_tokens
    """
    def __init__(self, num_tokens, num_hiddens, step, bidirectional, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnnlayer = nn.RNN(num_tokens, num_hiddens)
        self.num_hiddens = num_hiddens
        self.step = step
        self.num_tokens = num_tokens
        if bidirectional:
            self.num_directions = 2
            self.linear = nn.Linear(num_hiddens*2, num_tokens)
        
        else:
            self.num_directions = 1
            self.linear = nn.Linear(num_hiddens, num_tokens)

    def forward(self, input: torch.Tensor, state):
        X = F.one_hot(input.T.long(), self.num_tokens)
        X = X.to(torch.float32)
        Y, state = self.rnnlayer(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin(self, device, batch_size=1) -> torch.Tensor:
        if not isinstance(self.rnnlayer, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnnlayer.num_layers,
                                batch_size, self.num_hiddens), device=device)
        
        else:
            return (torch.zeros((self.num_directions*self.rnnlayer.num_layers,
                                batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions*self.rnnlayer.num_layers,
                                batch_size, self.num_hiddens), device=device))


if __name__ == "__main__":
    step=5
    hiddens=256
    token_size=27
    batch_size=10
    net = RNNModel(token_size, hiddens, step, False)
    features = torch.ones((batch_size, step))
    print(net(features, net.begin('cpu', batch_size=batch_size)))