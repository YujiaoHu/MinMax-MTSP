import torch
from torch.nn import functional as F
from .gnn_layer import etype_net
from .gnn_layer import gnn_op
from .gnn_layer import res_gnn_layer
from .gnn_layer import dim_change_conv
from .basic.utils import get_edge_feature
from .basic.utils import get_nn_node_feature


class agent_select(torch.nn.Module):
    def __init__(self, hidden_size, etypes, anum):
        self.anum = anum
        super(agent_select, self).__init__()

        self.anum = anum
        self.etypeNet = etype_net(etypes[1], etypes[0])
        layers = len(hidden_size) - 1
        self.embedding = []
        before_gnn = True
        for l in range(layers):
            if hidden_size[l] == hidden_size[l + 1]:
                mp_nn = res_gnn_layer(anum,
                                      nin=hidden_size[l],
                                      nout=hidden_size[l + 1],
                                      nedge_types=etypes[1],
                                      hop=False)
                before_gnn = False
            else:
                mp_nn = dim_change_conv(anum, hidden_size[l], hidden_size[l + 1], withrelu=before_gnn)
            self.add_module("mp_nn_{}".format(l), mp_nn)
            self.embedding.append(mp_nn)

    def forward(self, pts, nn_idx):
        # pts: [batch, f, n]
        # nn_idx : [batch, n, k]
        batch_size = pts.size(0)
        nnode = pts.size(2)
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: batch * input_feature_size * city_num * knn_k
        efeature = get_edge_feature(pts_knn, pts)  # efeature: batch * input_feature_size * city_num * knn_k
        etype = self.etypeNet(efeature)
        etype = etype.unsqueeze(1)
        etype = etype.repeat(1, self.anum, 1, 1, 1)
        nn_idx = nn_idx.unsqueeze(1)
        nn_idx = nn_idx.repeat(1, self.anum, 1, 1)

        nfeature = pts.unsqueeze(3).unsqueeze(1)
        nfeature = nfeature.repeat(1, self.anum, 1, 1, 1)

        for m in self.embedding:
            nfeature = m(nfeature, nn_idx, etype)
        return nfeature


class masked_graph_embedding(torch.nn.Module):
    def __init__(self, anum, insize, outsize):
        super(masked_graph_embedding, self).__init__()
        self.anum = anum
        self.etypeNet = etype_net(8, insize)
        self.gnn = gnn_op(anum=1, nin=insize, nout=outsize, nedge_types=8, hop=False)
        self.etype = None

    def forward(self, pts, nn_idx, nstep):
        # pts:[batch, fsize, nnodes]
        # nn_idx:[batch, nnodes, k]
        if nstep == 0:
            pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: batch * input_feature_size * city_num * knn_k
            efeature = get_edge_feature(pts_knn, pts)  # efeature: batch * input_feature_size * city_num * knn_k
            etype = self.etypeNet(efeature)
            etype = etype.unsqueeze(1)
            self.etype = etype.repeat(1, self.anum, 1, 1, 1)
        nn_idx = nn_idx.unsqueeze(1)
        nn_idx = nn_idx.repeat(1, self.anum, 1, 1)
        nfeature = pts.unsqueeze(3).unsqueeze(1)
        nfeature = nfeature.repeat(1, self.anum, 1, 1, 1)
        nfeature = self.gnn(nfeature, nn_idx, self.etype)
        return nfeature


class graph_embedding_v1(torch.nn.Module):
    def __init__(self, hidden_size, etypes, anum, common_hop=True, global_hop=True):
        super(graph_embedding_v1, self).__init__()
        # hidden_size : list, like [2, 16, 16, 32, 32, 64, 64]
        # etypes: list, including [infeature_size, outfeature_size]
        self.anum = anum
        self.etypeNet = etype_net(etypes[1], etypes[0])
        layers = len(hidden_size)-1
        self.embedding = []
        for l in range(layers):
            if hidden_size[l] == hidden_size[l+1]:
                mp_nn = res_gnn_layer(anum,
                                    nin=hidden_size[l],
                                    nout=hidden_size[l+1],
                                    nedge_types=etypes[1],
                                    hop=common_hop)
            else:
                mp_nn = dim_change_conv(anum, hidden_size[l], hidden_size[l+1])
            self.add_module("mp_nn_{}".format(l), mp_nn)
            self.embedding.append(mp_nn)

        # global pooling embedding
        self.global_embed = gnn_op(anum,
                              nin=hidden_size[-1] * (anum + 1),
                              nout=hidden_size[-1],
                              nedge_types=etypes[1],
                              hop=global_hop)

    def forward(self, pts, nn_idx):
        # pts: [batch, f, n]
        # nn_idx : [batch, n, k]
        batch_size = pts.size(0)
        nnode = pts.size(2)
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: batch * input_feature_size * city_num * knn_k
        efeature = get_edge_feature(pts_knn, pts)  # efeature: batch * input_feature_size * city_num * knn_k
        etype = self.etypeNet(efeature)
        etype = etype.unsqueeze(1)
        etype = etype.repeat(1, self.anum, 1, 1, 1)
        nn_idx = nn_idx.unsqueeze(1)
        nn_idx = nn_idx.repeat(1, self.anum, 1, 1)

        nfeature = pts.unsqueeze(3).unsqueeze(1)
        nfeature = nfeature.repeat(1, self.anum, 1, 1, 1)

        for m in self.embedding:
            nfeature = m(nfeature, nn_idx, etype)
        # nfeature: [batch, anum, f, n, 1]
        glbfeature, _ = torch.max(nfeature, dim=3)
        glbfeature = glbfeature.squeeze(3).contiguous().view(batch_size, 1, -1, 1, 1)
        glbfeature = glbfeature.repeat(1, self.anum, 1, nnode, 1)
        nfeature = torch.cat([glbfeature, nfeature], dim=2)
        nfeature = self.global_embed(nfeature, nn_idx, etype)
        return nfeature

    def normalize_pred(self, feature):
        # feature: batch , f
        return F.normalize(feature, p=2, dim=1)


class graph_embedding_v2(torch.nn.Module):
    # v2： do not consider global feature of other agent while doing node feature embdedding
    # self.global_embed is different with v1
    def __init__(self, hidden_size, etypes, anum, common_hop=True, global_hop=True):
        super(graph_embedding_v2, self).__init__()
        # hidden_size : list, like [2, 16, 16, 32, 32, 64, 64]
        # etypes: list, including [infeature_size, outfeature_size]
        self.anum = anum
        self.etypeNet = etype_net(etypes[1], etypes[0])
        layers = len(hidden_size)-1
        self.embedding = []
        for l in range(layers):
            if hidden_size[l] == hidden_size[l+1]:
                mp_nn = res_gnn_layer(anum,
                                    nin=hidden_size[l],
                                    nout=hidden_size[l+1],
                                    nedge_types=etypes[1],
                                    hop=common_hop)
            else:
                mp_nn = dim_change_conv(anum, hidden_size[l], hidden_size[l+1])
            self.add_module("mp_nn_{}".format(l), mp_nn)
            self.embedding.append(mp_nn)

        # global pooling embedding
        self.global_embed = gnn_op(anum,
                              nin=hidden_size[-1] * 2,
                              nout=hidden_size[-1],
                              nedge_types=etypes[1],
                              hop=global_hop)

    def forward(self, pts, nn_idx):
        # pts: [batch, f, n]
        # nn_idx : [batch, n, k]
        batch_size = pts.size(0)
        nnode = pts.size(2)
        pts_knn = get_nn_node_feature(pts, nn_idx)  # pts_knn: batch * input_feature_size * city_num * knn_k
        efeature = get_edge_feature(pts_knn, pts)  # efeature: batch * input_feature_size * city_num * knn_k
        etype = self.etypeNet(efeature)
        etype = etype.unsqueeze(1)
        etype = etype.repeat(1, self.anum, 1, 1, 1)
        nn_idx = nn_idx.unsqueeze(1)
        nn_idx = nn_idx.repeat(1, self.anum, 1, 1)

        nfeature = pts.unsqueeze(3).unsqueeze(1)
        nfeature = nfeature.repeat(1, self.anum, 1, 1, 1)

        for m in self.embedding:
            nfeature = m(nfeature, nn_idx, etype)
        # nfeature: [batch, anum, f, n, 1]
        glbfeature, _ = torch.max(nfeature, dim=3, keepdim=True)
        # glbfeature = glbfeature.squeeze(3).contiguous().view(batch_size, 1, -1, 1, 1)
        glbfeature = glbfeature.repeat(1, 1, 1, nnode, 1)
        nfeature = torch.cat([glbfeature, nfeature], dim=2)
        nfeature = self.global_embed(nfeature, nn_idx, etype)
        return nfeature

    def normalize_pred(self, feature):
        # feature: batch , f
        return F.normalize(feature, p=2, dim=1)

