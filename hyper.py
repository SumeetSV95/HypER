import torch
from torch.nn.init import xavier_normal_
import numpy as np
from torch.nn import functional as F, Parameter


class HypER(torch.nn.Module):
    def __init__(self, data, nf=32, lf=3, d_e=5, d_r=5):
        super(HypER, self).__init__()
        self.E = data.entityEmbeddings
        self.R = data.relationEmbeddings
        self.data = data
        self.nf = nf
        self.lf = lf
        self.d_e = d_e
        self.d_r = d_r
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.nf)
        self.bn2 = torch.nn.BatchNorm1d(self.d_e)
        self.loss = torch.nn.BCELoss()
        self.register_parameter('b', Parameter(torch.zeros(len(data.entityToId.keys()))))
        fc1_length = self.nf * self.lf
        print(self.d_r)
        fc_length = 1 * (self.d_e - self.lf + 1) * self.nf
        self.fc1 = torch.nn.Linear(self.d_r, fc1_length)
        self.fc = torch.nn.Linear(fc_length, self.d_e)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1, r):
        e1 = e1.view(-1, 1, self.d_e)  # batch_size * 1 * d_e
        x = self.bn0(e1)
        k = self.fc1(r)  # batch_size * in_channels * nf * lf
        k = k.view(-1, 1, self.nf, self.lf)
        k = k.view(e1.size(0) * 1 * self.nf, 1, self.lf)
        x = x.permute(1, 0, 2)
        # print(e1.size(0))
        # print(x.shape, k.shape)
        x = F.conv1d(x, k, groups=e1.size(0))
        # print("this is x")
        # print(x.shape)
        x = x.view(e1.size(0),  self.nf, self.d_e - self.lf + 1)
        x = x.permute(0, 1, 2).contiguous()
        x = self.bn1(x)
        # print("this is x conti")
        # print(x)
        x = x.view(e1.size(0), -1)
        # print("this is last x")
        # print(x.shape)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        # print("og prediction")
        # print(pred)
        # pred = np.random.rand(e1.size(0), self.E.weight.size(0))

        return pred
