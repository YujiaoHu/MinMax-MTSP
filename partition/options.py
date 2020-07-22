import argparse
import os
import sys

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="model parameters")
    parser.add_argument('--anum', type=int, default=5, help="agent num")
    parser.add_argument('--cnum', type=int, default=50, help="city num, including depot")
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--iteration', type=int, default=1000000, help='maximum training iteration')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--clip', type=bool, default=True, help='do you want to clip gradient')
    parser.add_argument('--clip_norm', type=float, default=3.0)
    parser.add_argument('--trainIns', type=int, default=10, help='S-sample, S setting')
    parser.add_argument("--modelpath", default=os.path.join(os.getcwd(), "savemodel"))
    opts = parser.parse_args(args)
    return opts
