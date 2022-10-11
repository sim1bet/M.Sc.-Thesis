# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:04:48 2021

@author: smnbe
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import quad
from numba import jit


# Counter for the number of spikes
@jit
def counter_s(data, t, mx):
  # Creates a time partition for the given data according to the number of intervals t
  time_mesh = np.linspace(1, mx, num=t)

  spk_ctr, _ = np.histogram(data, time_mesh)

  return spk_ctr

# Counter for the instances with k spikes
@jit
def counter_k(spk_ctr):
  inst_ctr = []
  # While cycle that counts the number of bins within which the same number of spikes occurs
  while len(np.nonzero(spk_ctr!=-1)[0])>0:
    idx = np.nonzero(spk_ctr!=-1)[0][0]
    # Number of spikes
    val = spk_ctr[idx]
    # Number of time bins having number of spikes "val"
    n_val = len(np.nonzero(spk_ctr==val)[0])
    
    inst_ctr.append(np.array([val, n_val], dtype=np.int64))

    spk_ctr[spk_ctr==val]=-1
  
  return inst_ctr


# Computation of resolution points
@jit
def hs(spk_ctr, spk_tot):
  # information extracted at different resolutions
  h_s=0
  for spk in spk_ctr:
    if spk!=0:
      h_s -= (1/np.log(spk_tot))*(spk/spk_tot)*np.log(spk/spk_tot)
    else:
      h_s = h_s
  

  return h_s


# Computation of relevance points
@jit
def hk(inst_ctr, spk_tot):
  # relevance of the information extracted for different resolutions
  h_k = 0
  for inst in inst_ctr:
    if inst[0]!=0:
      h_k -=(1/np.log(spk_tot))*(inst[0]*inst[1]/spk_tot)*np.log(inst[0]*inst[1]/spk_tot)
    else:
      h_k = h_k
  

  return h_k

@jit
def time_delay(data):
    # Computation of the time delay between subsequent spikes for the given data
    spk_idx = np.nonzero(data)[0]
    l = len(spk_idx)
    
    time_delay = []
    
    for i in range(l-1):
        time_delay.append(spk_idx[i+1]-spk_idx[i])
        
    return np.array(time_delay)


# Main for the computation of MSR
@jit
def MSR(mx, Pos, time_steps):
  # Number of the neuron (counter)
  n_neu = 1
  MSRs = []
  Main_Ne={}

  
  if type(Pos)==dict:
    L = len(Pos)
  else:
    L = len(Pos[:,0])


  for cnt in range(L):
    # Definition of structures for the scoring of the relevant measures
    Neu = {}
    H_S = []
    H_K = []

    if type(Pos)==dict:
      data = Pos[str(cnt+1)]
      # Total number of spikes in the current time series
      spk_tot = len(data)
    else:
      data = Pos[cnt,:]
      # Total number of spikes in the current time series
      spk_tot = np.count_nonzero(data)
    
    if spk_tot>5:
      for t in time_steps:

        # Definition of the counters (spikes, bins)
        spk_ctr = counter_s(data, t, mx)
        spk_hold = np.copy(spk_ctr)
        inst_ctr = counter_k(spk_hold)

        # Resolution and Relevance measure
        h_s = hs(spk_ctr, spk_tot)
        h_k = hk(inst_ctr, spk_tot)

        H_S.append(h_s)
        H_K.append(h_k)
  
      H_S.append(1)
      H_K.append(0)

      f = interp1d(H_S, H_K, kind='linear')
      MSR = quad(f, 0, 1)

      MSRs.append(MSR[0])
      
      tm_delay = time_delay(data)

      name = 'Neuron_'+str(n_neu)
      Neu['Neuron'] = n_neu
      Neu["H[S]"] = H_S
      Neu["H[K]"] = H_K
      Neu["MSR"] = MSR
      Neu["NeuRef"] = [spk_tot, cnt]
      Neu["TimeDelay"] = tm_delay

      Main_Ne[name]=Neu

      n_neu += 1

  return Main_Ne, MSRs



##############################################################################
# Functions for plotting of the results
def ranking(MSRs, Main_Ne, typ):
  plt.rcParams.update({'font.size': 22})
  # Creation of a copy of MSRs for ordering and indexing
  temp = np.argsort(MSRs)
  for j in range(1):
    name = 'Neuron_'+str(temp[-j-1]+1)
    txt = 'MSR neuron_'+str(Main_Ne[name]['Neuron'])+'_'+typ
    val = 'MSR: '+str(round(Main_Ne[name]['MSR'][0],4))
    # Visual info
    f = interp1d(Main_Ne[name]['H[S]'], Main_Ne[name]['H[K]'], kind='linear')
    ax = np.linspace(0, 1, 1000)

    
    plt.figure(figsize=(10,8))
    plt.plot(Main_Ne[name]['H[S]'], Main_Ne[name]['H[K]'], "o", ax, f(ax), "-", label=val)
    plt.legend()
    plt.fill_between(Main_Ne[name]['H[S]'], Main_Ne[name]['H[K]'], color="thistle")
    plt.xlabel('H[S]')
    plt.ylabel('H[K]')
    plt.savefig(txt)
    plt.show()
    plt.close()
    
# Distribution of MSR values for the subset of units considered
def histo(MSRs, txt):
  # Function for plotting an histogram with the values of the MSR
  # Visualization of their distribution for the given units
  plt.rcParams.update({'font.size': 22})

  #res = [20, 40, 60, 80, 100]
  res = [80]

  for r in res:
    TXT = txt + 'res_' + str(r)
    plt.figure(figsize=(12,8))
    plt.hist(MSRs, bins=r)
    plt.xlabel('MSR')
    plt.ylabel('N. of units')
    plt.savefig(TXT)
    plt.show()
    plt.close()
