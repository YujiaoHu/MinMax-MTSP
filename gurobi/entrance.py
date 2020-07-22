from VRPCenter import VRPCenter
import random
import os
import torch
import sys
import time
import os


def main_minmax(argv):
    city_num = int(argv[1])
    deliver_num = int(argv[2])
    datanum = int(argv[3])
    timeLimitation = int(argv[4])
    filepath = os.path.join(os.getcwd(), '../dataset')
    try:
        os.stat(filepath)
    except:
        os.mkdir(filepath)
    filepath = os.path.join(filepath, 'gurobi')
    try:
        os.stat(filepath)
    except:
        os.mkdir(filepath)
    filepath = os.path.join(filepath, 'TL{}'.format(timeLimitation))
    try:
        os.stat(filepath)
    except:
        os.mkdir(filepath)
    filepath = os.path.join(filepath, 'agent{}'.format(deliver_num))
    try:
        os.stat(filepath)
    except:
        os.mkdir(filepath)
    filepath = os.path.join(filepath, 'city{}'.format(city_num))
    try:
        os.stat(filepath)
    except:
        os.mkdir(filepath)
    filename = 'minmax_agent{}_city{}_num{}.pt'.format(deliver_num, city_num, datanum)
    filesave = os.path.join(filepath, filename)
    print(filesave)
    trainset = []
    starttime = time.time()
    vrp = VRPCenter(city_num, deliver_num)
    path = vrp.start(timeLimitation)
    endtime = time.time()
    train = {"cities": vrp.cities, "path": path, "time": endtime - starttime}
    # print(train)
    trainset.append(train)
    torch.save(trainset, filesave)
    print("save finished")


if __name__ == '__main__':
    main_minmax(sys.argv)

