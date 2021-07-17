# %% load dataset
import collections
import re
import torch
import torch.nn as nn
import requests
from utils.drawer import Drawer
from rnnmodel import RNNModel


def load_timemachine():
    file_path = r'./data/timemachine.txt'
    import os
    if not os.path.exists(file_path):
        url = r'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
        print(f'Downloading file from {url}')
        r = requests.get(url, stream=True, verify=True)
        os.makedirs(r'./data', exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(r.content)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# text = load_timemachine()
# print(text[0])
# %%


def tokenize(text, token_type='words'):
    if token_type == 'words':
        return [word for line in text for word in line]
    else:
        token = ''
        for line in text:
            token += line
            token += ' '
        return token


class Vocabulary:
    '''

    '''

    def __init__(self, words, min_freq=0) -> None:
        self.min_freq = min_freq
        self.counter = collections.Counter(words)
        self.sorted_words = sorted(self.counter.items(), key=lambda x: x[1])
        self.index_to_word = {k: v[0] for k, v in enumerate(self.sorted_words)}
        self.word_to_index = {v[0]: k for k, v in enumerate(self.sorted_words)}
        self.corpus = [self.word_to_index[i] for i in words]

    def __getitem__(self, index):
        if type(index) == str:
            return [self.word_to_index[i] for i in index]
        if isinstance(index, (list, tuple)):
            return ''.join([self.index_to_word[int(i)] for i in index])
        return self.corpus[index]

    def __len__(self):
        return len(self.corpus)

    def token_size(self):
        return len(self.sorted_words)


def get_corpus_dataset(corpus, step, length=-1):
    if length <= 0 or length > len(corpus)-step:
        length = len(corpus) - step
    features = torch.zeros((length, step))
    labels = torch.zeros((length, step))
    for i in range(step):
        features[:, i] = torch.Tensor(corpus[i: length+i])
        labels[:, i] = torch.Tensor(corpus[i+1: length+i+1])
    return torch.utils.data.TensorDataset(features, labels)
# %%
# voca = Vocabulary(tokenize(load_timemachine(), 'ch'))
# dataset = get_corpus_dataset(voca, 10)


def train(net, dataset, vocab, batch_size, lr, num_epochs, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    drawer = Drawer(figname='RNN', xlim=[0, num_epochs], ylim=[
                    0, 3], legend=['perplexity'])

    for epoch in range(num_epochs):
        perplexity = train_epoch(
            net, dataset, criterion, updater, batch_size, device)
        drawer.add(epoch, perplexity)
        prediction = predict(
            net, vocab, 'there happened this strange thing', 60, device)
        print(
            f'epoch:{epoch}, perplexity:{perplexity}, prediction:{prediction}')
    return net


def train_epoch(net, dataset, criterion, updater, batch_size, device):
    dataiter = iter(torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True))

    total, total_loss = 0., 0.
    state = net.begin(device, batch_size=batch_size)
    for x, y in dataiter:
        y = y.T.reshape(-1)
        x, y = x.to(device), y.to(device)
        state.detach_()
        y_hat, state = net(x, state)
        loss = criterion(y_hat, y.long()).mean()
        updater.zero_grad()
        loss.backward()
        updater.step()
        total += y.numel()
        total_loss += loss*y.numel()
    return (total_loss / total).cpu()


def run():
    lr = 0.5
    batch_size = 200
    step = 10
    num_hiddens = 256
    device = 'cuda:0'
    vocab = Vocabulary(tokenize(load_timemachine(), 'ch'))
    dataset = get_corpus_dataset(vocab, step)
    net = RNNModel(
        num_tokens=vocab.token_size(),
        num_hiddens=num_hiddens,
        step=step,
        bidirectional=False
    )
    net = net.to(device)
    train(net, dataset, vocab, batch_size, lr, num_epochs=500, device=device)
    return net, vocab


def predict(net: RNNModel, vocab, prefix, pred_length, device, state=None):
    if state is None:
        state = net.begin(device)
    tokens = vocab[prefix]
    for i in range(len(tokens)):
        inputs = torch.Tensor(tokens).reshape((1, -1))
        inputs = inputs.to(device)
        _, state = net(inputs, state)
    for i in range(pred_length):
        inputs = torch.Tensor(tokens).reshape((1, -1))
        inputs = inputs.to(device)
        y, state = net(inputs, state)
        y = int(y.argmax(1)[-1])
        tokens.append(y)
    return vocab[tokens]


# %%
run()

# %%
