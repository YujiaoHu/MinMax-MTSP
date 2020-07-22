# decoding part

import torch
from torch.nn import functional as F
import math
import os

class attention_decoding(torch.nn.Module):
    def __init__(self, fsize, anum):
        super(attention_decoding, self).__init__()
        self.anum = anum
        self.decoder = []
        self.glb_fixed_Q = []
        self.node_fixed_K = []
        self.node_fixed_V = []
        self.node_fixed_logit_K = []
        for a in range(self.anum):
            glb_embedding = torch.nn.Linear(self.anum * fsize + fsize, fsize)
            node_embedding = torch.nn.Conv2d(fsize, 3 * fsize, 1, 1)
            last_current_embedding = torch.nn.Conv2d(2 * fsize, fsize, 1, 1)
            project_out = torch.nn.Linear(fsize, fsize)

            self.add_module('glb_embedding_{}'.format(a), glb_embedding)
            self.add_module('node_embedding_{}'.format(a), node_embedding)
            self.add_module('last_current_{}'.format(a), last_current_embedding)
            self.add_module('project_out_{}'.format(a), project_out)

            self.decoder.append({'glb_embedding_{}'.format(a): glb_embedding,
                                 'node_embedding_{}'.format(a): node_embedding,
                                 'last_current_{}'.format(a): last_current_embedding,
                                 'project_out_{}'.format(a): project_out})
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, nfeature):
        batch_size = nfeature.size(0)

        self.glb_fixed_Q = []
        self.node_fixed_K = []
        self.node_fixed_V = []
        self.node_fixed_logit_K = []

        glbfeature, _ = torch.max(nfeature, dim=3)
        glbfeature = glbfeature.squeeze(3)
        glbfeature = glbfeature.contiguous().view(batch_size, -1)

        # glbfeature concat with depot feature
        for a in range(self.anum):
            # nfeature: [batch, anum, f, n, 1]
            deglb = torch.cat([glbfeature, nfeature[:, a, :, 0, 0]], dim=1)
            self.glb_fixed_Q.append(self.decoder[a]['glb_embedding_{}'.format(a)](deglb))
            self.glb_fixed_Q[a] = self.glb_fixed_Q[a].unsqueeze(2)

            x, y, z = self.decoder[a]['node_embedding_{}'.format(a)](nfeature[:, a, :, 1:]).chunk(3, dim=1)
            self.node_fixed_K.append(x)
            self.node_fixed_V.append(y)
            self.node_fixed_logit_K.append(z)
            self.node_fixed_K[a] = self.node_fixed_K[a].squeeze(3)
            self.node_fixed_V[a] = self.node_fixed_V[a].squeeze(3)
            self.node_fixed_logit_K[a] = self.node_fixed_logit_K[a].squeeze(3)

        result = []
        for a in range(self.anum):
            context_Q = self.glb_fixed_Q[a]
            # print("context_Q = ", context_Q, "self.node_fixed_K[a]", self.node_fixed_K[a])
            ucj = torch.bmm(context_Q.transpose(1, 2), self.node_fixed_K[a]) / math.sqrt(self.node_fixed_K[a].size(1))
            # temp_mask = mask.clone().unsqueeze(1)
            # ucj[temp_mask] = -math.inf
            new_context = torch.bmm(F.softmax(ucj, dim=2), self.node_fixed_V[a].transpose(1, 2))
            new_context = self.decoder[a]['project_out_{}'.format(a)](new_context.squeeze(1)).unsqueeze(1)
            logits = torch.bmm(new_context, self.node_fixed_logit_K[a]) / math.sqrt(self.node_fixed_K[a].size(1))
            logits = logits.squeeze(1) # logits: [batch, n]
            # print("att_decoingd part: before tanh", logits)
            logits = torch.tanh(logits) * 10
            result.append(logits)
        result = torch.stack(result, dim=2)
        result = self.softmax(result)
        return result

class prob_decoding(torch.nn.Module):
    def __init__(self, fsize, anum):
        super(prob_decoding, self).__init__()
        self.anum = anum
        self.decoder = []
        for a in range(self.anum):
            decoding = torch.nn.Sequential(
                torch.nn.Conv2d(fsize, 64, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 16, 1, 1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 1, 1, 1)
            )
            self.add_module('decode_{}'.format(a), decoding)
            self.decoder.append(decoding)

    def forward(self, nfeature, glbfeature):
        # decoding part
        # glbfeature:[batch, f, 1]
        batch_size = nfeature.size(0)
        nnode = nfeature.size(3)

        # nfeature: [batch, anum, f, n, 1]
        glbfeature = glbfeature.contiguous().view(batch_size, 1, -1, 1, 1)
        glbfeature = glbfeature.repeat(1, self.anum, 1, nnode, 1)
        nfeature = torch.cat([glbfeature, nfeature], dim=2)

        result = []
        for a in range(self.anum):
            m = self.decoder[a]
            r = m(nfeature[:, a])  # r: [batch, 1, n, 1]
            r = self.normalize_pred(r.squeeze(3).squeeze(1))
            result.append(r)
        result = torch.stack(result, dim=1)  # result: [batch, anum, n]
        return nfeature, result

    def normalize_pred(self, feature):
        # feature: batch , f
        return F.normalize(feature, p=2, dim=1)