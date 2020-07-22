import torch
from enum import Enum
# try:
#     from encoding.nn import SyncBatchNorm
# except:
SyncBatchNorm = torch.nn.BatchNorm2d

class mp_conv_type(Enum):
    NO_EXTENSION = 0
    ORIG_WITH_NEIGHBOR = 1
    ORIG_WITH_DIFF = 2


class pw_gnn_op(torch.nn.Module):
    def __init__(self,
                 nin,
                 nou,
                 nedge_types,
                 bias=True,
                 bn=True,
                 extension=mp_conv_type.ORIG_WITH_DIFF,
                 activation_fn='relu',
                 aggregtor='max'):
        super(pw_gnn_op, self).__init__()

        self.nin = nin
        self.nou = nou
        self.nedge_types = nedge_types
        self.extension = extension

        self.filters1 = torch.nn.Parameter(
            torch.zeros(nin, nou, nedge_types, dtype=torch.float32))
        self.filters2 = torch.nn.Parameter(
            torch.zeros(nin, nou, nedge_types, dtype=torch.float32))
        self.filters = [self.filters1, self.filters2]
        for f in self.filters:
            f.data.uniform_(-0.01, 0.01)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(nou))
            self.bias.data.uniform_(0, 0.05)
        else:
            self.bias = None

        if bn:
            self.bn = SyncBatchNorm(nou)
        else:
            self.bn = None
        if isinstance(activation_fn, torch.nn.Module):
            self.activation_fn = activation_fn
        elif activation_fn == 'relu':
            self.activation_fn = torch.nn.ReLU(inplace=True)
        else:
            self.activation_fn = None

        if isinstance(aggregtor, str):
            if aggregtor == 'max':

                def agg_max(x):
                    res, *_ = torch.max(x, dim=3, keepdim=True)
                    return res

                self.aggregtor = agg_max
            elif aggregtor == 'mean':
                self.aggregtor = lambda x: torch.mean(x, dim=3, keepdim=True)

        else:
            self.aggregtor = aggregtor

    def to_edge_feature(self, node_feature, nn_idx):
        batch_size = nn_idx.shape[0]
        node_feature = node_feature.squeeze()  # shape n x b x c
        if batch_size == 1:
            node_feature = node_feature.unsqueeze(0)
        # print(node_feature.shape)
        # print(nn_idx.shape)
        assert (batch_size == node_feature.shape[0])
        npts = nn_idx.shape[1]
        assert (npts == node_feature.shape[1])
        k = nn_idx.shape[2]

        nidx = nn_idx.view(batch_size, -1).unsqueeze(2).repeat(
            1, 1, node_feature.shape[2])  # shape n x b x k

        # print(node_feature.shape)
        # print(nidx.shape)

        pts_knn = node_feature.gather(1, nidx).view(batch_size, npts, k, -1)

        return pts_knn

    def forward(self, x, nn_idx, etype):
        # x: batch * input_feature_size * city_num * 1
        batch_size = x.shape[0]
        nin = x.shape[1]
        nnodes = x.shape[2]
        k = nn_idx.shape[2]
        nedge_type = etype.permute(0, 2, 3, 1).contiguous().view(
            -1, self.nedge_types, 1)
        if self.extension == mp_conv_type.NO_EXTENSION:
            node_feature = x.permute(0, 2, 3, 1).contiguous().view(
                batch_size * nnodes, nin)
            node_feature = node_feature.mat_mul(
                self.filters.view(nin, self.nou * self.nedge_types)).view(
                    batch_size, nnodes, self.nou * self.nedge_types)
            edge_feature = self.to_edge_feature(node_feature, nn_idx).view(
                -1, self.nou, self.nedge_types)

            edge_feature = edge_feature.bmm(nedge_type).view(
                batch_size, nnodes, k, self.nou)

        else:
            node_feature = x.permute(0, 2, 3, 1).contiguous().view(
                batch_size * nnodes, nin)
            # node_feature: [batch * city_num, 6]

            nfeature = node_feature.matmul(self.filters[0].view(
                nin, self.nou * self.nedge_types)).view(
                    batch_size, nnodes, 1, self.nou * self.nedge_types)
            # nfeature: [batch, city, 1, 1024]
            efeature = node_feature.matmul(self.filters[1].view(
                nin, self.nou * self.nedge_types)).view(
                    batch_size, nnodes, 1, self.nou * self.nedge_types)
            # efeature: [batch, city, 1, 1024]
            efeature_nn = self.to_edge_feature(efeature, nn_idx)
            # efeature_nn: [batch, city, knn_k, 1024]

            if self.extension == mp_conv_type.ORIG_WITH_NEIGHBOR:
                edge_feature = nfeature + efeature_nn
            elif self.extension == mp_conv_type.ORIG_WITH_DIFF:
                edge_feature = nfeature + efeature - efeature_nn

            else:
                raise ValueError("self.extension must be one of mp_conv_type")

            edge_feature = edge_feature.view(
                -1, self.nou, self.nedge_types).bmm(nedge_type).view(
                    batch_size, nnodes, k, self.nou)
        nfeature = edge_feature.permute(0, 3, 1, 2).contiguous()
        if self.bias is not None:
            # print(nfeature.shape)
            # print(self.bias.shape)
            nfeature = nfeature + self.bias.view(1, self.nou, 1, 1)

        if self.aggregtor is not None:
            nfeature = self.aggregtor(nfeature)
            # print(nfeature.shape)

        if self.bn is not None:
            nfeature = self.bn(nfeature)

        if self.activation_fn is not None:
            nfeature = self.activation_fn(nfeature)

        return nfeature