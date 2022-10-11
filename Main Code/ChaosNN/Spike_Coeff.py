#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:52:47 2021

@author: sim1bt
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numba import jit

#############################################################################
# The computation of the following coefficients follows the eqautions outlined
# in https://doi.org/10.1371/journal.pone.0178062
#############################################################################

@jit
def LocalVar(un_time_int):
    
    # Computation of the local variation coefficient
    L_v = 0
    L = len(un_time_int)
       
    for i in range(L-1):
        if (un_time_int[i+1]+un_time_int[i])!=0:
            L_v += (3/(L-1))*((un_time_int[i]-un_time_int[i+1])/(un_time_int[i+1]+un_time_int[i]))**2
        
    return L_v

@jit
def burst(un_time_int):
    
    # Computation of the burstiness coefficient
    time_d_mean = np.mean(un_time_int)
    time_d_std = np.std(un_time_int)
    
    if (time_d_mean+time_d_std)!=0:
        b = (time_d_std-time_d_mean)/(time_d_std+time_d_mean)
    else:
        b = -1
    
    return b

@jit
def memory(un_time_int):
    
    # Computation of the memory coefficient
    L = len(un_time_int)
    
    # Mean and Standard deviation excluding the last inter-time
    m_1 = np.mean(un_time_int[:-1])
    sig_1 = np.std(un_time_int[:-1])
    
    # Mean and Standard deviation excluding the first inter-time
    m_2 = np.mean(un_time_int[1:])
    sig_2 = np.std(un_time_int[1:])
    
    # Memory Coefficient
    M = 0
    if sig_1 != 0 and sig_2 != 0:
        for i in range(L-1):
            M += 1/(L-1)*((un_time_int[i]-m_1)*(un_time_int[i+1]-m_2))/(sig_1*sig_2)
    else:
        M = 1
        
    return M

#############################################################################
# Main for the computation of the coefficients for all the units
#############################################################################
@jit
def Coef_Series(Main_Ne):
    
    # Definition of the dictionary {unit: [L_v, b, M]}
    Coef = {}
    
    ky = Main_Ne.keys()
    for un in ky:
        L_v = LocalVar(Main_Ne[un]['TimeDelay'])
        b = burst(Main_Ne[un]['TimeDelay'])
        M = memory(Main_Ne[un]['TimeDelay'])
        
        Coef[un] = [L_v, b, M]
    
    return Coef
    
#############################################################################
# Post processing function for the extraction of coefficients for those units
# for which a MSR value was computed at that specific current
#############################################################################
@jit
def extract_coef(Main_Ne, Coef):
    
    # Parsing of the dictionary Main_Ne to assess the units for which a MSR 
    # value has been computed
    
    LocVar = []
    Burst = []
    Mem = []
    MSR = []
    
    it = Coef.keys()
    for un in it:
        LocVar.append(Coef[un][0])
        Burst.append(Coef[un][1])
        Mem.append(Coef[un][2])
        MSR.append(Main_Ne[un]['MSR'][0])
        
    LocVar = np.array(LocVar)
    Burst = np.array(Burst)
    Mem = np.array(Mem)
    MSR = np.array(MSR)
    
    return LocVar, Burst, Mem, MSR

#############################################################################
# Function for the plotting of the relations
# MSR-Burstiness, LocVar-Burstiness
# MSR-Memory, LocVar-Memory
# LocVar-MSR
# as done by Cubero arXiv:1802.10354
#############################################################################

def plot_spk_coef_2D(coef1, coef2, com, xlm, ylm, cmlm, xlb, ylb, title, txt):
    
    # Update of the font size
    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(15,10))
    plt.scatter(coef1, coef2, c=com, cmap = matplotlib.cm.plasma)
    # Setting colormap scale
    plt.clim(cmlm[0], cmlm[1])
    # Setting axis limits
    plt.xlim(xlm[0], xlm[1])
    plt.ylim(ylm[0], ylm[1])
    # Setting axis labels
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    # Setting plot title and saving
    cbar = plt.colorbar()
    cbar.ax.set_title("Firing Rate")
    plt.title(title)
    plt.savefig(txt)
    plt.show()
    plt.close()

def plot_dist(meas, xlabl, title):
  # Function for plotting an histogram with the values of the measure
  # Visualization of their distribution for the given units
  plt.rcParams.update({'font.size': 22})

  #res = [20, 40, 60, 80, 100]
  res = [80]

  for r in res:
    TXT = title + 'res_' + str(r)
    TXT = TXT.replace('.','_')
    plt.figure(figsize=(12,8))
    plt.hist(meas, bins=r)
    plt.xlabel(xlabl)
    plt.ylabel('N. of units')
    plt.title(title)
    plt.savefig(TXT)
    plt.show()
    plt.close()
    
