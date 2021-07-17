# %%
import torch
from torch import nn
from argparse import Namespace
import matplotlib.pyplot as plt


args = Namespace(
    T=1000,
    tau=64,
    lr=0.1,
    batch_size=16,
    num_epoch=30,
    n_train=600
)
time = torch.arange(1, args.T+1, dtype=torch.float32)
y = torch.sin(0.01*time) + torch.normal(0, 0.1, (args.T,))
plt.plot(time, y)
plt.show()

# %%
# choose tau features by order
features = torch.zeros((args.T-args.tau, args.tau))
net = nn.Sequential(
    nn.Linear(args.tau, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

def init_param(m:nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_param)
for i in range(args.tau):
    features[:, i] = y[i:args.T-args.tau+i] # 看得懂不？是不是又忘了
labels = y[args.tau:].reshape((-1,1))
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()


for epoch in range(args.num_epoch):
    optimizer.zero_grad()
    y_hat = net(features[:600])
    loss = criterion(y_hat, labels[:600])
    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch}, loss: {loss}')
# %% onestep prediction
onestep_preds = net(features)
with torch.no_grad():
    plt.plot(time, y)
    plt.plot(time[args.tau:], onestep_preds)
# %% multistep prediction
multi_preds = torch.zeros([args.T-args.tau, 1])
multi_preds[:600] = onestep_preds[:600]

for i in range(600, args.T-args.tau):
    multi_preds[i] = net(multi_preds[i-args.tau: i].reshape(-1, args.tau))
with torch.no_grad():
    plt.plot(time, y, label='groundtruth')
    plt.plot(time[args.tau:], onestep_preds, label='onestep')
    plt.plot(time[args.tau:], multi_preds, label='multistep')
    plt.legend()

# %%
 