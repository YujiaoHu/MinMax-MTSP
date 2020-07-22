import torch
from enum import Enum
# try:
#     from encoding.nn import SyncBatchNorm
# except:
SyncBatchNorm = torch.nn.BatchNorm2d


class hop_gnn_op(torch.nn.Module):
    def __init__(self,
                 nin,
                 nou):
        super(hop_gnn_op, self).__init__()

        self.nin = nin
        self.nou = nou

        self.tfilter = torch.nn.Parameter(
            torch.zeros(1, nin, nou, dtype=torch.float32))
        self.tfilter.data.uniform_(-0.01, 0.01)

        self.nfilter = torch.nn.Parameter(
            torch.zeros(1, nin, nou, dtype=torch.float32))
        self.nfilter.data.uniform_(-0.01, 0.01)

    def forward(self, nfeature, tfeature):
        # nfeature:[batch, nin, n, 1]
        # tfeature:[batch, nin, n, 1]

        # batch_size, nnode = nfeature.size(0), nfeature.size(2)
        # mask = torch.ones(batch_size, nnode, self.nou).to(nfeature.device)
        # mask[:, 0] = 0
        batch_size = nfeature.size(0)
        nfeature = nfeature.squeeze(3).permute(0, 2, 1)
        tfeature = tfeature.squeeze(3).permute(0, 2, 1)
        # nfeature = nfeature + mask * (torch.bmm(nfeature, self.nfilter.repeat(batch_size, 1, 1))\
                   # + torch.bmm(tfeature, self.tfilter.repeat(batch_size, 1, 1)))
        nfeature = torch.bmm(nfeature, self.nfilter.repeat(batch_size, 1, 1))\
                   + torch.bmm(tfeature, self.tfilter.repeat(batch_size, 1, 1))
        return nfeature.permute(0, 2, 1).unsqueeze(3)