import torch
import multiprocessing
from .ortool_entrance import entrance as ortools
import gc

def classification_samping(probs):
    # print(probs)
    batch_size, anum, cnum = probs.size(0), probs.size(1), probs.size(2)
    sample_probs = probs.permute(0, 2, 1).contiguous().view(batch_size * cnum, anum)
    idxs = sample_probs.multinomial(1).squeeze(1).view(batch_size, cnum)
    gather_prob = torch.gather(probs, dim=1, index=idxs.unsqueeze(1))
    gather_prob = torch.log(gather_prob)
    sum_log_probs = torch.sum(gather_prob.squeeze(1), dim=1)
    return idxs, sum_log_probs, anum


def classify_tourform(inputs):
    cf_tour = inputs[0]
    cf_coords = inputs[1]
    cf_coords = cf_coords[:, :2]
    anum = inputs[2]
    tourlen = torch.zeros(anum)
    tourset = []
    buildTour = False
    for a in range(anum):
        atour = cf_tour[a]
        if 0 in atour:
            x = atour.long()
        else:
            steps = atour.size(0)
            x = torch.zeros(steps + 1).long()
            x[1:] = atour
        # print(x)
        assert torch.sum(x.eq(0)) == 1
        xcoord = torch.gather(cf_coords, 0, x.unsqueeze(1).repeat(1, 2))
        # use ortool compute, return tour length
        # print("starting planning, ", x.size()[0])
        tourlen[a], singleTour = ortools(xcoord)
        # print(tourlen[a])

        # if buildTour:
        #     cnum = len(singleTour)
        #     for c in range(cnum):
        #         singleTour[c] = x[singleTour[c]]
        tourset.append(singleTour)
    return [tourlen, tourset]


def classification_tourform(tour, coords, anum):
    pool = multiprocessing.Pool(processes=26)

    device = tour.device
    batch_size, cnum, dims = coords.size()
    number_samples = tour.size(1)

    samples_tours = [[[] for n in range(number_samples)] for b in range(batch_size)]
    multi_idxs = []
    for b in range(batch_size):
        for n in range(number_samples):
            agent_tour = []
            for a in range(anum):
                aidxs = tour[b, n].eq(a)
                atour = torch.nonzero(aidxs).view(-1).cpu() + 1
                agent_tour.append(atour)
                # print("b = {}, n = {}, a = {}, atour = {}".format(b, n, a, atour))
                samples_tours[b][n].append(torch.tensor(atour).to(device))
            # print("agent tour", agent_tour)
            multi_idxs.append([agent_tour, coords[b].cpu(), anum])

    result = pool.map(classify_tourform, multi_idxs)
    pool.close()
    pool.join()

    tourlen = []
    tourset = []
    for b in range(batch_size):
        nsampleTour = []
        for n in range(number_samples):
            tourlen.append(result[b*number_samples+n][0])
            nsampleTour.append(result[b*number_samples+n][1])
        tourset.append(nsampleTour)
    tourlen = torch.stack(tourlen, dim=0).view(batch_size, number_samples, anum)

    # # single process
    # test_result = []
    # for b in range(batch_size):
    #     inputs = [index[b], coords[b], anum]
    #     test_result.append(classify_tourform(inputs))
    # test_result = torch.stack(test_result, dim=0)
    # print(" test result:", test_result-result)
    # print("result:", result[0])
    return tourlen.clone(), tourset

def get_tour_len(tour, coords, anum):
    device = coords.device
    batch_size, number_samples, cnum_1 = tour.size()

    t = anum * number_samples
    bsize = 3000 // t
    ntimes = batch_size // bsize
    if batch_size % bsize > 0:
        ntimes += 1

    tourlen = []
    samples_tours = []
    for n in range(ntimes):
        endbatch = min(  n * bsize + bsize, batch_size  )
        # print("---------------------------------", n, endbatch)
        tempTour, tempSampleTours = classification_tourform(tour[n*bsize:endbatch], coords[n*bsize:endbatch], anum)
        # print("tempTour", tempTour.size())
        tourlen.append(tempTour)
        samples_tours = samples_tours + tempSampleTours
    # tourlen, samples_tours = classification_tourform(tour, coords, anum)
    tourlen = torch.cat(tourlen, dim=0)
    # print("tourlen.size = ", tourlen.size())
    return tourlen.to(device), samples_tours


def get_tour_len_singleTrack(tour, coords, anum):
    device = tour.device
    batch_size, cnum, dims = coords.size()
    number_samples = tour.size(1)

    # samples_tours = [[[] for n in range(number_samples)] for b in range(batch_size)]
    tourlen = []
    tourset = []
    for b in range(batch_size):
        nsampleTour = []
        for n in range(number_samples):
            agent_tour = []
            # print("partition:", tour[b,n])
            for a in range(anum):
                aidxs = tour[b, n].eq(a)
                atour = torch.nonzero(aidxs).view(-1).cpu() + 1
                agent_tour.append(atour)
                # print("b = {}, n = {}, a = {}, atour = {}".format(b, n, a, atour))
                # samples_tours[b][n].append(torch.tensor(atour).to(device))
            # print("agent tour", agent_tour)
            result = classify_tourform([agent_tour, coords[b].cpu(), anum])
            tourlen.append(result[0])
            nsampleTour.append(result[1])
        tourset.append(nsampleTour)
    tourlen = torch.stack(tourlen, dim=0).view(batch_size, number_samples, anum)
    return tourlen.to(device), tourset
