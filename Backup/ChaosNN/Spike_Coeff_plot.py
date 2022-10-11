#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:33:20 2021

@author: sim1bt
"""

import numpy as np
from Spike_Coeff import Coef_Series, extract_coef, plot_spk_coef_2D, plot_dist
from Stat_utilities import extract_units_idx

def spike_plot_cur(Stat, idx, idx_tlt, name, T):
    
    # Computation of the Local Variation Coefficient, Burstiness Coefficient
    # and Memory Coefficient
    # N.B. some values of the memory coefficient are not displayed, as the time
    # intervals, having all integer values, could potentially generate a zero variance
    # vector
    spk_tot, unit_idx = extract_units_idx(Stat[idx]['Main_Ne'])
    Coef = Coef_Series(Stat[idx]['Main_Ne'])
    # Extraction of the vectors for the plotting
    LocVar, Burst, Mem, MSR = extract_coef(Stat[idx]['Main_Ne'], Coef)
    # Mean Vector of the metrics
    Mean = [np.mean(LocVar), np.mean(Burst), np.mean(Mem)]
    # Setting of the labels and axis limits
    # Local Variation Coefficient
    LocVarlm = [np.amin(LocVar),np.amax(LocVar)]
    LocVarlb = "Coefficient Local Variation"
    # Burstiness Coefficient
    Burstlm = [-1, 1]
    Burstlb = "Burstiness Coefficient"
    # Memory Coefficient
    Memlm = [-1, 1]
    Memlb = "Memory Coefficient"
    # MSR coefficient
    MSRlm = [0,0.35]
    MSRlb = "MSR"
    # Definition of the color map
    # Firing rate
    cm = spk_tot/T
    cmlim = [0,1]

    # Plotting of the distribution of firing rates and of different spike measures
    # Resolution: 80 bins

    # Firing rate distribution
    title_fire = "Firing Rate Distr. I = "+idx_tlt
    x_fire = "Firing Rate"
    plot_dist(cm, x_fire, title_fire)

    # Burstiness Distribution
    title_brst = "Burstiness Distr. I = "+idx_tlt
    x_brst = "Burstiness Coef."
    plot_dist(Burst, x_brst, title_brst)

    # Memory Distribution
    title_mem = "Memory Distr. I = "+idx_tlt
    x_mem = "Memory Coef."
    plot_dist(Mem, x_mem, title_mem)

    # Plotting of the correlation between different spike measures
    # 3rd axis : firing rate
    
    titleMSR_Burst = "MSR - Burstiness Plot I = "+idx_tlt
    saveMSR_Burst = "MSR - Burstiness Plot I = "+name
    plot_spk_coef_2D(MSR, Burst, cm, MSRlm, Burstlm, cmlim, MSRlb, Burstlb, titleMSR_Burst, saveMSR_Burst)

    titleMSR_Mem = "MSR - Memory Plot I = "+idx_tlt
    saveMSR_Mem = "MSR - Memory Plot I = "+name
    plot_spk_coef_2D(MSR, Mem, cm, MSRlm, Memlm, cmlim, MSRlb, Memlb, titleMSR_Mem, saveMSR_Mem)

    titleLV_Burst = "Local Variation - Burstiness Plot I = "+idx_tlt
    saveLV_Burst = "Local Variation - Burstiness Plot I = "+name
    plot_spk_coef_2D(LocVar, Burst, cm, LocVarlm, Burstlm, cmlim, LocVarlb, Burstlb, titleLV_Burst, saveLV_Burst)

    titleLV_Mem = "Local Variation - Memory Plot I = "+idx_tlt
    saveLV_Mem = "Local Variation - Memory Plot I = "+name
    plot_spk_coef_2D(LocVar, Mem, cm, LocVarlm, Memlm, cmlim, LocVarlb, Memlb, titleLV_Mem, saveLV_Mem)

    titleLV_MSR = "Local Variation - MSR Plot I = "+idx_tlt
    saveLV_MSR = "Local Variation - MSR Plot I = "+name
    plot_spk_coef_2D(LocVar, MSR, cm, LocVarlm, MSRlm, cmlim, LocVarlb, MSRlb, titleLV_MSR, saveLV_MSR)

    return Mean
