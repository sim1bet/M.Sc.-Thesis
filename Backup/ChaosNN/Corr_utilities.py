#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:26:09 2021

@author: sim1bt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from Stat_utilities import * 
from numba import jit

############################################################################
# Function for the computation of the correlation between two given vector 
# valued quantities across different input conditions (i_e, i_i)
############################################################################
@jit
def corr_graph_MSR(Stat, graph_stat, Curr, I_E, I_I, vlist, k, txt):
    
    #Time 
    T = np.shape(Curr[0])[1]
    
    #Creation of blank templates for comparison purposes
    best_stat_MSR = np.zeros((k,len(I_E)))
    corr = np.zeros((1,len(I_E))).squeeze()
    corr_best = np.zeros((1,len(I_E))).squeeze()
    
    cnt = 0
    
    for (i_e,i_i) in zip(I_E,I_I):
        
        idx = '('+str(i_e)+','+str(i_i)+')'
        idx_tlt = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        
        # Extracts the indices of the units for which MSR has been computed
        # Not necessarily all the units, given the constraint
        # spk_tot>1
        Main_Ne = Stat[idx]['Main_Ne']
        spk_tot, units_idx = extract_units_idx(Main_Ne)
        
        graph_stat_ref = graph_stat[units_idx]
        
        # Consideration of the best k nodes according to graph_stat
        best_stat_idx = np.argsort(graph_stat_ref)[-k:]
        best_stat = np.sort(graph_stat_ref)[-k:]
        
        # Selection of the currents according to best_stat_idx
        best_curr = np.zeros((len(best_stat_idx), T, len(Curr)))
        cr_cnt = 0
        for cr in Curr:
            best_curr[:,:,cr_cnt] = cr[units_idx[best_stat_idx],:]
            cr_cnt += 1
        
        # Loading of MSR values for given current and extraction of values as
        # as dictated by best_stat_idx
        MSR = np.array(Stat[idx]['MSRs'])
        MSR_best = MSR[best_stat_idx]

        # Plotting of the point correlation between the graph measure and MSR
        stat_MSR_plot(graph_stat_ref, spk_tot, MSR, txt, idx_tlt)
        
        # Update of blank template for plotting of I-MSR for the units selected
        # by best_stat_idx
        best_stat_MSR[:,cnt] = MSR_best
        
        MSR_corr = pearsonr(MSR,graph_stat_ref)
        MSR_corr_best = pearsonr(MSR_best, best_stat)
       
        corr[cnt] = MSR_corr[0]
        corr_best[cnt] = MSR_corr_best[0] 
        cnt += 1
        
    cur_vec = (I_E+I_I)*0.5
    # Plotting of the MSR-GM correlation
    corr_plot(corr, corr_best, cur_vec, txt, k)
    # Plotting of the MSR trajectory across currents for the selected units
    I_MSR_plot(best_stat_idx, best_stat_MSR, cur_vec, txt)
    
    # Plotting of the incoming currents to the units selected by the graph measure
    l_cur = len(Curr)
    #curr_plot(best_curr, best_stat_idx, T, txt, l_cur)
    
############################################################################
# Function for the extraction and plotting of a quantity f() 
# for the units presenting the lowest variance across input conditions (i_e, i_i)
############################################################################
@jit
def low_var_MSR(Stat, I_E, I_I, n_nod, k):
    ###################################################
    # Construction of the matrix containing the MSR values 
    # for each incoming current
    ###################################################
    MSR_cur = np.zeros((n_nod,len(I_E)))
    
    c = 0
    
    for (i_e,i_i) in zip(I_E,I_I):
        idx = '('+str(i_e)+','+str(i_i)+')'
        
        # Extracts the indices of the units for which MSR has been computed
        # Not necessarily all the units, given the constraint
        # spk_tot>1
        Main_Ne = Stat[idx]['Main_Ne']
        spk_tot, units_idx = extract_units_idx(Main_Ne)
        
        # assignment of the MSR values to the corresponding matrix column
        MSR_cur[units_idx,c] = Stat[idx]['MSRs']
        c += 1
    
    # Computation of the unit-specific MSR mean and variance vector
    mean_MSR, var_MSR = stat(MSR_cur)
    idx_mean_pos = np.nonzero(mean_MSR>0)[0]
    
    # Extraction of the variances relevant for the ordering
    var_clean = var_MSR[idx_mean_pos]
    
    idx_small_var = np.argsort(var_clean)
    best_idx_small_var = []
    i = 0
    while k>0:
        if np.mean(MSR_cur[idx_mean_pos[idx_small_var[i]],:])>0.09:
            best_idx_small_var.append(idx_mean_pos[idx_small_var[i]])
            k -= 1
        i += 1
        
    best_idx_small_var = np.array(best_idx_small_var)
    
    np.save("SmalVarIdx.npy",best_idx_small_var)
    
    smal_var_MSR = MSR_cur[best_idx_small_var,:]
    
    # Plotting of the I-MSR plot for units with smallest MSR variance across
    # currents
    cur_vec = (I_E+I_I)*0.5
    txt = "Smallest Variance"
    I_MSR_plot(best_idx_small_var, smal_var_MSR, cur_vec, txt)
    
###########################################################################
# Function for the plotting of the MSR against Log(M), where M is the total 
# number of spikes in the time series
###########################################################################
@jit
def MSR_logM(Stat, I_E, I_I, n_nod):
    
    # Creation of the mean current vector
    cur_vec = (I_E+I_I)*0.5
    # Maximum of the mean current for plotting purposes
    m_cur = np.amax(cur_vec)
    
    # maximum logM among all possible
    m_logM = 0
    
    Plot_val = {}
    
    # Generation of the MSR-Log(spk_tot) vectors
    for (i_e,i_i) in zip(I_E,I_I):
        idx = '('+str(i_e)+','+str(i_i)+')'
        idx_tlt = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        MSR = Stat[idx]['MSRs']
        
        # Extraction of the indices of the units for which MSR was computed
        spk_tot, units_idx = extract_units_idx(Stat[idx]['Main_Ne'])
        logM = np.log(spk_tot)
        if np.amax(logM) > m_logM:
            m_logM = np.amax(logM)

        # Data points for excitatory population
        exc_idx = np.nonzero(units_idx<1000)[0]
        logM_exc = np.zeros((len(exc_idx),))
        MSR_exc = np.zeros((len(exc_idx),))
        for exc in range(len(exc_idx)):
            logM_exc[exc] = logM[exc_idx[exc]]
            MSR_exc[exc] = MSR[exc_idx[exc]]
        # Data points for the inhibitory population
        inh_idx = np.nonzero(units_idx>=1000)[0]
        logM_inh = np.zeros((len(inh_idx),))
        MSR_inh = np.zeros((len(inh_idx),))
        for inh in range(len(inh_idx)):
            logM_inh[inh] = logM[inh_idx[inh]]
            MSR_inh[inh] = MSR[inh_idx[inh]]
        
        Plot_val[idx_tlt] = [[logM_exc, MSR_exc],[logM_inh, MSR_inh]]

    plt.rcParams.update({'font.size': 22})
        
    for (i_e,i_i) in zip(I_E,I_I):
        
        idx = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        name = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        name = name.replace('.','_')
        
        txt = "LogM-MSR plot I = "+idx
        tlt = "LogM-MSR plot current_ "+name
        
        plt.figure(figsize=(15,10))
        plt.scatter(Plot_val[idx][0][0], Plot_val[idx][0][1], color="red", label="Excitatory Units")
        plt.scatter(Plot_val[idx][1][0], Plot_val[idx][1][1], color="green", label="Inhibitory Units")
        # Setting of the scaling for the axis
        plt.xlim(0, m_logM)
        plt.ylim(0,0.35)
        # Setting of the axis labels
        plt.xlabel("log(M)")
        plt.ylabel("MSR")
        plt.legend()
        plt.title(txt)
        plt.savefig(tlt)
        plt.close()
        
    cnt = 0
    ax = plt.figure(figsize=(25,20)).add_subplot(projection="3d")
    for (i_e,i_i) in zip(I_E,I_I):
        
        idx = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        p = ax.scatter(xs=Plot_val[idx][0][0], ys=Plot_val[idx][0][1], zs=cur_vec[cnt], zdir="y", color="red")
        q = ax.scatter(xs=Plot_val[idx][1][0], ys=Plot_val[idx][1][1], zs=cur_vec[cnt], zdir="y", color="green")
        cnt += 1

    # Definition of the scaling for the axis
    ax.set_xlim(0,m_logM)
    ax.set_ylim(0,m_cur)
    ax.set_zlim(0,0.35)
    # setting of axis label
    ax.set_xlabel("log(M)")
    ax.set_ylabel("I")
    ax.set_zlabel("MSR")
    plt.legend()
    plt.savefig("MSR-LogM Plot")
    plt.close()
    
############################################################################
# Function iterating the graph analysis over the entire set of graph
# theoretical measures provided
# N.B. requires to fix the number k of best units to analyze
############################################################################

def graph_cor(graph_stat, graph_txt, Stat, Curr, I_E, I_I):
    ###################################################
    # Computation and plotting of the different correlation for all the types 
    # of graph statistics
    ###################################################
    
    # Number of best statistics to gather from graph_stat_instance
    k = 5
    
    # Number of units and list of units for selection of indices from the 
    # MSR list 
    n_nod = np.shape(graph_stat[0])[0]
    vlist = np.arange(n_nod)
    
    # Extraction of the units with smallest MSR variance
    low_var_MSR(Stat, I_E, I_I, n_nod, k)
    
    # MSR-LogM plot across currents
    MSR_logM(Stat, I_E, I_I, n_nod)

    for i in range(len(graph_stat)):
        corr_graph_MSR(Stat, graph_stat[i], Curr, I_E, I_I, vlist, k, graph_txt[i])
