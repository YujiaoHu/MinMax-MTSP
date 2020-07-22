from .vrp import entrance
import torch
import time
import os
import sys

recordfile = os.path.join(os.getcwd(), 'timeRecord.txt')

def time_recoder():
    if not os.path.exists(recordfile):
        f = open(recordfile, 'w+')
        f.write("record ORTool running time")
        f.close()

    for cnum in range(20, 500):
        f = open(recordfile, 'a+')
        f.write('\n\n')
        f.close()
        for anum in range(2, 10):
            start_time = time.time()
            for num in range(200):
                entrance(cnum, anum)
            end_time = time.time()
            f = open(recordfile, 'a+')
            f.write("\ncnum = {}, anum = {}, average time = {}".format(cnum, anum, (end_time - start_time)/num))
            f.close()
            print("\n cnum = {}, anum = {}, average time = {}".format(cnum, anum, (end_time - start_time)/num))

def prepare_dir(objective, anum, cnum):
    datapath = os.path.join(os.getcwd(), "../../dataset")
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "ortools")
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, objective)
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "agent{}".format(anum))
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "city{}".format(cnum))
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)
    return datapath

def generate_ortools_data(objective, anum, cnum, dnum):
    datapath = prepare_dir(objective, anum, cnum)
    filename = os.path.join(datapath,
                            "minmax_ortools_{}_agent{}_city{}_num{}.pt"
                            .format(objective, anum, cnum, dnum))
    start_time = time.time()
    tourlen, coords = entrance(cnum, anum)
    duration = time.time() - start_time
    dataset = {'cities': coords, 'tourlen': tourlen, 'time': duration}
    torch.save(dataset, filename)
    # print("save ", filename)
