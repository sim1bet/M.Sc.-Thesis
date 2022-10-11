#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:12:56 2021

@author: sim1bt
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

###########################################################################
# Function for the extraction of indices and total number of spikes
###########################################################################
@jit
def extract_units_idx(Main_Ne):
    
    extr_idx = []
    spk_tot = []
    
    it = Main_Ne.keys()
    for name in it:
        spk_tot.append(Main_Ne[name]['NeuRef'][0])
        extr_idx.append(Main_Ne[name]['NeuRef'][1])
    
    return np.array(spk_tot, dtype=np.int), np.array(extr_idx, dtype=np.int)

###########################################################################
# Function for the computation of mean and variance of individual rows
# in a matrix
###########################################################################
@jit
def stat(mat):
    # Requires a d-tensor as input
    # d > 1
    # Compuation of the unit mean MSR values across currents
    mean = np.mean(mat,axis=1)
    
    # Construction of the variance(current) matrix
    var = np.var(mat, axis=1)
    
    return mean, var

###########################################################################
# Functions for the plotting of correlations and currents
###########################################################################

# Correlation between MSR and graph measure

def corr_plot(corr, corr_best, cur_vec, txt, k):
    # Setting of axis labels and title for the graphs
    ylab = "MSR-"+txt+" Correlation"
    title = "Correlation between MSR and "+txt
    ylab_best = "Best "+str(k)+" MSR-"+txt+" Correlation" 
    title_best = "Best "+str(k)+" (according to "+txt+") MSR-"+txt

    plt.rcParams.update({'font.size': 22})
    
    # Plotting of the correlation for all the units    
    plt.figure(figsize=(15,10))
    plt.plot(cur_vec, corr, linestyle="solid", color="green", marker="x")
    plt.xlabel("I (Average current to network)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(ylab)
    plt.show()
    plt.close()
    
    # Plotting of the correlation for the units classified as best
    # according to the given graph measure
    plt.figure(figsize=(15,10))
    plt.plot(cur_vec, corr_best, linestyle="solid", color='blue', marker="o")
    plt.xlabel("I (Average current to network)")
    plt.ylabel(ylab)
    plt.title(title_best)
    plt.savefig(ylab_best)
    plt.show()
    plt.close()
    
# MSR across external currents for a specified unit

def I_MSR_plot(idx, stat_MSR, cur_vec, txt):

    plt.rcParams.update({'font.size': 22})

    # Plotting of the MSR values across currents for selected units
    for i in range(len(idx)):
        un = "MSR unit: "+str(idx[i])
        tlt = "I-MSR plot unit: "+str(idx[i])+", stat: "+txt
        save_tlt = "I-MSR plot unit_"+str(idx[i])+" stat_"+txt
        
        plt.figure(figsize=(15,10))
        plt.plot(cur_vec, stat_MSR[i,:], linestyle="-.", color="purple", marker="v")
        plt.ylim((0,0.35))
        plt.xlabel("I (Average current to network)")
        plt.ylabel(un)
        plt.title(tlt)
        plt.savefig(save_tlt)
        plt.show()
        plt.close()
        
# Incoming currents to a specific set of neurons, across external currents
        
def curr_plot(best_curr, best_stat_idx, T, txt, l):

    plt.rcParams.update({'font.size': 22})

    # Plotting of the incoming currents to a set of selected units
    for j in range(l):
        for c in range(len(best_stat_idx)):
            cr = 6*j
            txt_jc = "Current to unit "+str(best_stat_idx[c])+" for I="+str(cr)+" input, stat: "+txt
            save_txt_jc = "Current to unit "+str(best_stat_idx[c])+" for I="+str(cr)+" input stat_"+txt
            
            plt.figure(figsize=(15,10))
            plt.plot(range(T), best_curr[c,:,j], color="red", linestyle="solid") 
            plt.xlabel("Time")
            plt.ylabel("I Current")
            plt.title(txt_jc)
            plt.savefig(save_txt_jc)
            plt.show()
            plt.close()


def stat_MSR_plot(stat_gr, spike_tot, MSR, txt, idx):

    plt.rcParams.update({'font.size': 22})

    # Definition of the graph title and and the file name
    name = idx
    name = name.replace('.','_')

    title_M = "MSR-"+txt+" correlation: I = "+idx
    save_title_M = "MSR-"+txt+" correlation: I = "+name

    title_S = "N. Spikes-"+txt+" correlation: I = "+idx
    save_title_S = "N Spikes-"+txt+" correlation: I = "+name

    # Plotting of the point correlation between the two metrics
    plt.figure(figsize=(15,10))
    plt.plot(stat_gr, MSR, marker="o", linestyle="")
    #plt.clim(0,0.35)
    plt.xlabel(txt)
    plt.ylabel('MSR')
    #plt.colormap()
    plt.title(title_M)
    plt.savefig(save_title_M)
    plt.show()
    plt.close()

    # Plotting of the point correlation between the two metrics
    plt.figure(figsize=(15,10))
    plt.plot(stat_gr, spike_tot, marker="o", linestyle="", color="green")
    #plt.clim(0,0.35)
    plt.xlabel(txt)
    plt.ylabel('N. Spikes')
    #plt.colormap()
    plt.title(title_S)
    plt.savefig(save_title_S)
    plt.show()
    plt.close()
