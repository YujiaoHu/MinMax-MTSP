import torch
from .basic.hop_gnn_operation import hop_gnn_op as hoplayer
from .basic.pw_gnn_operation import pw_gnn_op as pwlayer


class etype_net(torch.nn.Module):
    def __init__(self, nedge_type, nfeature_size=3):
        super(etype_net, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(nfeature_size, nedge_type * 2, 1, 1),
            torch.nn.BatchNorm2d(nedge_type * 2), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(nedge_type * 2, nedge_type, 1, 1, bias=False))

    def forward(self, edge_feature):
        return self.main(edge_feature)


class res_gnn_layer(torch.nn.Module):
    def __init__(self, anum, nin, nout, nedge_types, hop=True):
        super(res_gnn_layer, self).__init__()
        self.layer = gnn_op(anum, nin, nout, nedge_types, hop)

    def forward(self, nfeature, nn_idx, etype):
        return self.layer(nfeature, nn_idx, etype) + nfeature


class dim_change_conv(torch.nn.Module):
    def __init__(self, anum, nin, nout, withrelu=True):
        super(dim_change_conv, self).__init__()
        self.anum = anum
        self.conv = []
        if withrelu is True:
            for a in range(self.anum):
                m = torch.nn.Sequential(
                    torch.nn.Conv2d(nin, nout, 1, 1),
                    torch.nn.BatchNorm2d(nout),
                    torch.nn.ReLU(inplace=True)
                )
                self.add_module('conv_{}'.format(a), m)
                self.conv.append(m)
        else:
            for a in range(self.anum):
                m = torch.nn.Sequential(
                    torch.nn.Conv2d(nin, nout, 1, 1)
                )
                self.add_module('conv_{}'.format(a), m)
                self.conv.append(m)

    def forward(self, nfeature, nn_idx, etype):
        # nfeature: [batch, anum, nin, n, 1]
        result = []
        for a in range(self.anum):
            m = self.conv[a]
            r = m(nfeature[:, a])
            result.append(r)
        return torch.stack(result, dim=1)


class gnn_op(torch.nn.Module):
    def __init__(self,
                 anum,
                 nin,
                 nout,
                 nedge_types,
                 hop=True):
        super(gnn_op, self).__init__()
        self.anum = anum
        self.hop = hop
        self.pw_gnn_part = []
        for a in range(self.anum):
            pw_op = pwlayer(nin, nout, nedge_types)
            self.add_module('pw_{}'.format(a), pw_op)
            self.pw_gnn_part.append(pw_op)
        if hop:
            self.hop_gnn_part = []
            for a in range(self.anum):
                hop_op = hoplayer(nout, nout)
                self.add_module('hop_{}'.format(a), hop_op)
                self.hop_gnn_part.append(hop_op)

    def forward(self, nfeature, nn_idx, etype):
        # nfeature: [batch, anum, nin, n, 1]
        # nn_idx: [batch, anum, n, k]
        # etype: [batch, anum, nedge_types, n, k]

        pw_result = []
        for a in range(self.anum):
            m = self.pw_gnn_part[a]
            r = m(nfeature[:, a], nn_idx[:, a], etype[:, a])
            pw_result.append(r)
        nfeature = torch.stack(pw_result, dim=1)
        if self.hop is True:
            hop_result = []
            mutul_or_meg = torch.sum(nfeature, dim=1)
            for a in range(self.anum):
                m = self.hop_gnn_part[a]
                r = m(nfeature=nfeature[:, a], tfeature=mutul_or_meg)
                hop_result.append(r)
            nfeature = torch.stack(hop_result, dim=1)
        return nfeature
