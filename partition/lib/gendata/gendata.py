import torch
from torch.utils.data import Dataset, DataLoader
import os
from .ortools_generator import generate_ortools_data


class gendata_from_gurobi(Dataset):
    def __init__(self, scale, anum, activate='train'):
        super(gendata_from_gurobi, self).__init__()
        self.scale = scale
        self.anum = anum
        self.k = 10
        self.activate = activate
        self.dataset = []
        self.datanum = 0
        if self.activate == 'test':
            datapath = os.path.join(os.getcwd(), '../dataset/gurobi/agent{}/city{}'.format(self.anum, self.scale))
            print(datapath)
            self.load_data(datapath)
            self.datanum = len(self.dataset)

    def __len__(self):
        if self.activate == 'train':
            return 10000000
        else:
            self.datanum = len(self.dataset)
            print("total number = ", self.datanum)
            return self.datanum

    def load_data(self, path):
        duration = 0.0
        gurobi_length = 0.0
        for i in range(1000):
            filename = "minmax_agent{}_city{}_num{}.pt".format(self.anum, self.scale, i)
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                data = torch.load(filepath)
                data = data[0]
                coord = (torch.tensor(data['cities']) + 1) / 2
                tour = data['path']
                if self.activate is not 'train':
                    tusage = data['time']
                    tourlen = torch.zeros(self.anum)
                    for a in range(self.anum):
                        tourlen[a] = self.computing_tourlen(coord, tour[a])
                    duration += tusage
                    gurobi_length += torch.max(tourlen)
                self.dataset.append([coord, tour])

        if self.activate is not 'train':
            self.datanum = len(self.dataset)
            if self.datanum <= 0:
                print("!!!!!!!!!!!error!!!!")
            else:
                print("\ngurobi testing dataset, anum = {}, cnum = {},\n\t\t "
                      "totoal testing instance = {},  \n\t\t"
                      "ave duration = {:4f}, ave length = {:4f}\n*****\n\n"
                      .format(self.anum, self.scale, self.datanum,
                              duration/self.datanum, gurobi_length/self.datanum))

    def __getitem__(self, idx):
        if self.activate == 'train':
            while True:
                success = True
                coord = torch.rand(self.scale, 2)
                for s in range(1, self.scale):
                    if torch.sum(torch.abs(coord[0] - coord[s])) < 1e-3:
                        success = False
                        break
                if success is True:
                    break
            knn = self.compute_knn(coord[:, :2])
            return coord, knn
        else:
            coord = self.dataset[idx][0]
            knn = self.compute_knn(coord[:, :2])
            tour = self.dataset[idx][1]
            tourlen = torch.zeros(self.anum)
            for a in range(self.anum):
                tourlen[a] = self.computing_tourlen(coord, tour[a])
            return coord, knn, tourlen

    def compute_knn(self, cities):
        city_square = torch.sum(cities**2, dim=1, keepdim=True)
        city_square_tran = torch.transpose(city_square, 1, 0)
        cross = -2 * torch.matmul(cities, torch.transpose(cities, 1, 0))
        dist = city_square + city_square_tran + cross
        knn = torch.argsort(dist, dim=-1)
        knn = knn[:, :self.k]
        return knn

    def computing_tourlen(self, coords, tour):
        atour = torch.tensor(tour)
        steps = atour.size(0)
        x = torch.zeros(steps + 1).long()
        y = x.clone()
        x[1:] = atour
        y[:steps] = atour
        xcoord = torch.gather(coords[:, :2], 0, x.unsqueeze(1).repeat(1, 2))
        ycoord = torch.gather(coords[:, :2], 0, y.unsqueeze(1).repeat(1, 2))
        temp = xcoord - ycoord
        return torch.sum(torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2))


class gendata_from_ortools(Dataset):
    def __init__(self, objective, scale, anum):
        super(gendata_from_ortools, self).__init__()
        self.scale = scale
        self.anum = anum
        self.k = 10
        self.dataset = []
        self.datanum = 0
        self.objective = objective

        datapath = os.path.join(os.getcwd(),
                                '../dataset/ortools/{}/agent{}/city{}'
                                .format(objective, self.anum, self.scale))
        print(datapath)
        self.datapath = datapath
        self.load_data(datapath)
        self.datanum = len(self.dataset)
        if self.datanum < 1:
            self.dataset = []
            print("... generting ortools test data...")
            for n in range(1000):
                generate_ortools_data(self.objective, self.anum, self.scale, n)
            print("finish generating data ")
            self.load_data(self.datapath)
            self.datanum = len(self.dataset)

    def __len__(self):
        self.datanum = len(self.dataset)
        return self.datanum

    def load_data(self, path):
        x = 0
        y = 0
        for i in range(1000):
            filename = "minmax_ortools_{}_agent{}_city{}_num{}.pt"\
                .format(self.objective, self.anum, self.scale, i)
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                data = torch.load(filepath)
                coord = torch.tensor(data['cities'])
                tourlen = data['tourlen']
                tusage = data['time']
                x += tusage
                self.dataset.append([coord, tourlen])
                y += torch.max(tourlen)
        print("\nortools testing dataset, anum = {}, cnum = {}".format(self.anum, self.scale))
        print("\t\ttime usage in the dataset is {}".format(x/len(self.dataset)))
        print("\t\taverage tour length is {}".format(y/len(self.dataset)))
        print("****\n\n")

    def __getitem__(self, idx):
        coord = self.dataset[idx][0]
        knn = self.compute_knn(coord[:, :2])
        tourlen = self.dataset[idx][1]
        return coord, knn, tourlen

    def compute_knn(self, cities):
        city_square = torch.sum(cities**2, dim=1, keepdim=True)
        city_square_tran = torch.transpose(city_square, 1, 0)
        cross = -2 * torch.matmul(cities, torch.transpose(cities, 1, 0))
        dist = city_square + city_square_tran + cross
        knn = torch.argsort(dist, dim=-1)
        knn = knn[:, :self.k]
        return knn

