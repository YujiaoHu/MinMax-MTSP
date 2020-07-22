import os
import torch
from options import get_options
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from lib.layers.partitionNet import mtsp
from lib.gendata.gendata import gendata_from_gurobi
from lib.gendata.gendata import gendata_from_ortools
# from lib.ortool.tourlen_computing import get_tour_len_singleTrack as orhelp
from lib.ortool.tourlen_computing import get_tour_len as orhelp
import time

def tourlen_computing(inputs, tour, anum):
    # inputs: coordinates [batch. cnum, 2]
    # tour: [batch, number_samples, cnum_1]
    with torch.no_grad():
        tourlen, samples_tours = orhelp(tour, inputs, anum)
    return tourlen, samples_tours


class TrainModleMTSP(nn.Module):
    def __init__(self, load=True,
                 objective='MinMax',
                 _modelpath=os.path.join(os.getcwd(), "../savemodel"),
                 anum=2,
                 cnum=20,
                 _device=torch.device('cuda:0'),
                 clip=True,
                 clip_argv=3,
                 lr=1e-5,
                 train_instance=10,
                 common_hop=False,
                 global_hop=False,
                 ):
        super(TrainModleMTSP, self).__init__()
        orhelper = True,

        self.load = load
        self.device = _device
        self.icnt = 0
        self.val = 0
        self.objective = objective
        self.modelpath = _modelpath
        self.anum = anum
        self.cnum = cnum
        self.orhelper = orhelper
        self.baseline = None
        self.lr = lr
        self.mgpu = False
        self.clip = clip
        self.clip_argv = clip_argv
        self.train_instance = train_instance

        self.valset = gendata_from_ortools('test', self.cnum, self.anum)
        if clip is True:
            self.model_name = "Partition_onegnn_tanhx_att_anum={}_cnum={}_lr={}_baseline=None_comhop={}_glohop={}_clip={}_trainIns={}" \
                .format(self.anum, self.cnum, self.lr, common_hop, global_hop, self.clip_argv, self.train_instance)
        else:
            self.model_name = "Partition_onegnn_tanhx_att_anum={}_cnum={}_lr={}_baseline=None_comhop={}_glohop={}_clip={}_trainIns={}" \
                .format(self.anum, self.cnum, self.lr, common_hop, global_hop, self.clip, self.train_instance)
        self.model = mtsp([2, 16, 16, 64, 64, 128, 128], [2, 16], anum, common_hop, global_hop)
        self.model.to(self.device)

        self.writer = SummaryWriter('../runs_anum={}_cnum={}/{}'.format(self.anum, self.cnum, self.model_name))
        self.modelfile = os.path.join(self.modelpath, '{}.pt'.format(self.model_name))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if load:
            print("loading model:{}".format(self.modelfile))
            if os.path.exists(self.modelfile):
                checkpoint = torch.load(self.modelfile, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.icnt = checkpoint['icnt'] + 1
                self.val = checkpoint['ival']
                print("Model loaded")
            else:
                print("No Model loaded")

    def return_model(self):
        return self.model

    def rl_loss_computing(self, logits, tourlen, partition):
        #  tourlen:[batch, number_samples, anum]
        #  partition:[batch, number_samples, cnum-1]
        #  logits:[batch, cnum-1, anum]
        maxlen = torch.max(tourlen, dim=2)[0]
        baselineTourlen = torch.mean(maxlen, dim=1, keepdim=True)
        advantage = maxlen - baselineTourlen  # advantage:[batch, number_samples]

        temp_partition = partition.permute(0, 2, 1)
        probsloss = torch.gather(logits, 2, temp_partition)  # [batch, cnum-1, number_samples]
        probsloss = torch.sum(torch.log(probsloss), dim=1)

        loss = torch.mean(probsloss * advantage)
        return loss

    def eval(self, source='gurobi', maxiter=20, batch_size=32):
        self.model.eval()
        print("testing with ortools ... ")
        loader = DataLoader(self.valset, batch_size=batch_size, shuffle=True, num_workers=8)
        maxiter = max(1, min(self.valset.datanum // batch_size, maxiter))

        net_len = []
        ortools_len = []
        replan_len = []

        with torch.no_grad():
            iterloader = iter(loader)
            for it in range(self.val, self.val + maxiter):
                try:
                    infeature, knn, target_tourlen = next(iterloader)
                except StopIteration:
                    print("StopIteration")
                    iterloader = iter(loader)
                    infeature, knn, target_tourlen = next(iterloader)

                infeature, knn, target_tourlen = \
                        infeature.to(self.device), knn.to(self.device), target_tourlen.to(self.device)
                start_time = time.time()
                probs, partition = self.model(infeature.permute(0, 2, 1), knn, maxsample=True, instance_num=1)
                net_tourlen, tours = tourlen_computing(infeature, partition, self.anum)
                net_tourlen = net_tourlen.view(batch_size, self.anum)

                # replan_net_tourlen, replan_net_duration = \
                #     orplanning_under_inital_solution(infeature, self.anum, 5, tours)

                # compare with ortools
                net_max, _ = torch.max(net_tourlen, dim=1)
                target_max, _ = torch.max(target_tourlen, dim=1)
                # replan_max, _ = torch.max(replan_net_tourlen, dim=1)

                net_max = net_max.view(-1)
                target_max = target_max.view(-1)
                # replan_max = replan_max.view(-1)

                net_len.append(net_max)
                ortools_len.append(target_max)
                # replan_len.append(replan_max)
                # print("time usage: ", (time.time() - start_time)/batch_size)

            net_len = torch.cat(net_len, dim=0)
            ortools_len = torch.cat(ortools_len, dim=0)
            # replan_len = torch.cat(replan_len, dim=0)

            net_or_diff = net_len / ortools_len
            print("***net_or_diff")
            print("mean: ", torch.mean(net_or_diff),
                  "max: ", torch.max(net_or_diff),
                  "min: ", torch.min(net_or_diff))
            print("----average")
            print("ortools mean: ", torch.mean(ortools_len),
                  "\nnet mean: ", torch.mean(net_len),
                  # "\nreplan mean: ", torch.mean(replan_len),
                  "\ngap: ", torch.mean(net_len) / torch.mean(ortools_len))
            print("-------------------------------\n")
            self.writer.add_scalar('val/net_or_gap', torch.mean(net_or_diff), it)
        print("testing ends")
        self.val = self.val + maxiter
        self.model.train()

    def train_without_baseline(self, maxiter=10000000, batch_size=32):
        train_set = gendata_from_gurobi(self.cnum, self.anum)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        iterloader = iter(loader)
        for it in range(self.icnt, maxiter):
            self.model.train()
            try:
                infeature, knn = next(iterloader)
            except StopIteration:
                print("StopIteration")
                iterloader = iter(loader)
                infeature, knn = next(iterloader)
            infeature, knn = infeature.to(self.device), knn.to(self.device)
            probs, partition = self.model(infeature.permute(0, 2, 1), knn,
                                          maxsample=False, instance_num=self.train_instance)
            tourlen, tours = tourlen_computing(infeature, partition, self.anum)
            loss = self.rl_loss_computing(probs, tourlen, partition)
            loss.backward()
            if self.clip is True:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_argv)
            self.optimizer.step()

            if it % 100 == 0:
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'icnt': it,
                            'ival': self.val
                            }, self.modelfile)
                print("------------------")
                print("saved model: iter = {}, modelfile = {}".format(it, self.modelfile))
                print("------------------")
            self.writer.add_scalar('train/loss', loss.item(), it)
            maxlen = torch.max(tourlen, dim=2)[0]
            minmaxlen = torch.min(maxlen, dim=1)[0]
            self.writer.add_scalar('train/max', torch.mean(minmaxlen), it)
            # print(torch.mean(torch.max(tourlen, dim=1)[0]))

            if it % 1000 == 0 and it > 0:
                self.eval(batch_size=128)


def main():
    opts = get_options()
    anum = opts.anum
    cnum = opts.cnum
    batch_size = opts.batch_size
    maxiter = opts.iteration
    device = torch.device(opts.cuda)
    lr = opts.lr
    clip = opts.clip
    clip_argv = opts.clip_norm
    trainIns = opts.trainIns
    modelpath = opts.modelpath

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    tsp = TrainModleMTSP(_modelpath=modelpath,
                         anum=anum,
                         cnum=cnum,
                         _device=device,
                         clip=clip,
                         clip_argv=clip_argv,
                         lr=lr,
                         train_instance=trainIns)
    tsp.eval(batch_size=batch_size)
    tsp.train_without_baseline(maxiter=maxiter, batch_size=batch_size)


if __name__ == '__main__':
    print(os.getcwd())
    # main_test()
    main()
