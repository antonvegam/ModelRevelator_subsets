#Function to compute column-wise summary on a multiple sequence alignment

import time
import subprocess
import sys
import os
import random
import math
import glob
import re

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)

samples = 500
block_taxa = 4
input_length_lstm = 10000
block_length = 80

def subsets(lines):
    """
    :param lines: Muliple sequence alignment to be preprocessed
    :return: A 200x200x17 tensor for the neural network
    """
    num_positions = block_length
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])
    msa = np.array([list(y) for y in raw_msa])
    
    def make_profile(m):

        mm = np.zeros([num_positions, 17], dtype=np.float16)
        for c in range(0, num_positions):

            substitutions_map = [
                'AA',
                'CC',
                'GG',
                'TT',
                'AC',
                'CG',
                'GT',
                'AT',
                'CT',
                'AG'
            ]

            base_map = {
                'A': 0,
                'C': 0,
                'G': 0,
                'T': 0
            }         

            for z in m[:, c]:
                base_map[z] += 1 #count number of different nucleotides for each column

            for count, tt in enumerate(substitutions_map):
                first, second = tt[0], tt[1]
                if first == second and base_map[first] > 1:
                    mm[c, count] = base_map[first] / block_length
                    
                elif first != second:
                    mm[c, count] = base_map[first] * base_map[second] / block_length
            
            for count, n in enumerate(base_map):
                mm[c, count + 12] += base_map[n]          
            
            for count, t in enumerate(mm[c,:]):
                if t != 0:
                    if (count == 9) or (count == 8):#check for transitions
                        mm[c,11] += 1
                    elif (count == 4) or (count == 5) or (count == 6) or (count == 7):#check for transversions
                        mm[c,10] += 1

        return mm
    try:
        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e, 'error when processing subsamples MSA')
        print('MSA shape:', msa.shape)

    summary_stats = []

    
    for sample_count in range(samples):
        rand_taxa = np.random.randint(0, msa.shape[0], size=block_taxa, dtype=int)
        rand_seq_start = np.random.randint(0, msa.shape[1], size=1, dtype=int)[0]

        if rand_seq_start + block_length >= msa_width:
            overflow_size = block_length - (msa_width - rand_seq_start)

            smpl = np.concatenate((msa[rand_taxa, rand_seq_start:msa_width], msa[rand_taxa, 0:overflow_size]), axis=1)           
            #sumary statistics for each subsample
            stats = make_profile(smpl)
            stats[:,16] = rand_seq_start
            summary_stats.append(stats)
        else:
            smpl = msa[rand_taxa, rand_seq_start:(rand_seq_start + block_length),]
                        
            #sumary statistics for each subsample
            stats = make_profile(smpl)
            summary_stats.append(stats)
            stats[:,16] = rand_seq_start

    sampled_msa = np.concatenate(summary_stats, axis=0)

    msa = np.reshape(sampled_msa, (200, 200, 17))

    return msa
