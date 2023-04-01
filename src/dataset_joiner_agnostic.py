import pandas as pd
import numpy as np
import argparse
import time
import pickle as pkl



parser = argparse.ArgumentParser(description='Compute features from sparse dataset for the agnostic setting')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')
parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')
parser.add_argument('--temporal',type=int,
                    help='Choice of time span. Choices 0,1,10,20,50,100?')

parser.add_argument('--cutoff',type=int,
                    help='Choice of cutoff. Choices 0,10,20,50,100,100?')

parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')

args = parser.parse_args('--inputdir /datanaco/dharp/compounds/datasets/  --outputdir ../datasets/compound_agnostic/ --temporal 0 --cutoff 0'.split())
