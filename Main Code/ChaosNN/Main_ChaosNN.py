# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:34:11 2021

@author: smnbe
"""

import numpy as np

from MSR import MSR, ranking, histo
from Network import ChaosNN
from Graph_Analysis import *
from Corr_utilities import graph_cor
from Spike_Coeff_plot import spike_plot_cur
from Pop_Stat import pop_act

# Definition of the parameters
N_E = 1000
N_I = 1000
theta_E = 1.0
theta_I = 0.8

# For Fixed weights
# J_EI = -2.0; J_II = -1.8
# For Random weights
# J_EI = -1.0; J_II = -1.0
J_EE = 1.0
J_IE = 1.0
J_EI = -2.0
J_II = -1.8

T = 60000

# definition of the time-meshes steps
time_steps_1 = np.linspace(2, 80, 40, dtype=np.int64)
time_steps_2 = np.linspace(100, 10000, 110, dtype=np.int64)
time_steps = np.concatenate((time_steps_1, time_steps_2))

# Empty lists for mean exc. and inh. activity
M_ex = []
M_in = []
# Empty list for Average MSR across currents
mean_MSR = []
# Empty list for Average Spike Metrics across current
mean_SPKM = []

# Empty list for the currents
Curr = []

# Span of currents to feed
I_E_v = np.linspace(1.35,14,30)
I_I_v = np.linspace(1,12.5,30)
I_F = zip(I_E_v,I_I_v)

# Condition for the weights w = {fixed,gamma}
w = "fixed"
# Condition for the input stoc = {True, False}
stoc = False

# Construction of the network 
Chaos = ChaosNN(N_E,N_I,theta_E, theta_I,J_EE,J_EI,J_IE,J_II,T, stoc)
Chaos.J_W_WattsStr(w)

iterat = np.arange(N_E+N_I)
np.random.shuffle(iterat)

Stat = {}

cn = 0
for (i_e,i_i) in I_F:
    
    idx = '('+str(i_e)+','+str(i_i)+')'
    rel = {}

    # Generation of the currents and of the activity
    Chaos.Current(i_e,i_i)
    Chaos.Prop_act_w(iterat)
    
    M_ex.append(Chaos.m_ex)
    M_in.append(Chaos.m_in)
    
    #Chaos.Select_sub()
    
    Main_Ne, MSRs = MSR(T, Chaos.Pos, time_steps)
    mean_MSR.append(np.mean(MSRs))

    rel['Main_Ne'] = Main_Ne
    rel['MSRs'] = MSRs

    Stat[idx] = rel
	
    if cn%6==0:
    
        Chaos.Select_sub()
        Chaos.plot_raster(i_e,i_i)
        
        Curr.append(Chaos.Curr)
        
        if T<3000:
            txt = "Curr_W = "+str(cn)+".npy"
            np.save(txt, Chaos.Curr)

    cn += 1

if T<3000:
    np.save('Exc_W.npy', M_ex)
    np.save('Inh_W.npy', M_in)
    np.save('I_exc_W.npy', I_E_v)
    np.save('I_Inh_W.npy', I_I_v)
    np.save('Rel_Stat_W.npy', Stat)
    
np.save('J_Coupl_W.npy',Chaos.J_W)
np.save('mean_MSR.npy', mean_MSR)

#Graph Analysis on J
#InDeg, Betw, Clos, Katz, LocCl, PgRk = graph_an(Chaos.J_W)

# Plotting of GraphMeasures-MSR correlations
#graph_stat = [InDeg, Betw, Clos, Katz, LocCl, PgRk]
#graph_txt = ["In Degree", "Betweenness Centrality", "Closeness Centrality", "Katz Centrality", "Local Clustering", "Pagerank"]

#graph_cor(graph_stat, graph_txt, Stat, Curr, I_E_v, I_I_v)

# Plotting of MSR distribution, BestMSR and time series coefficients
for (i_e, i_i) in zip(I_E_v,I_I_v):
    idx = '('+str(i_e)+','+str(i_i)+')'
    idx_tlt = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
    name = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
    name = name.replace('.','_')
    #ranking(Stat[idx]['MSRs'],Stat[idx]['Main_Ne'],name)
    #histo(Stat[idx]['MSRs'],name)
    Mean = spike_plot_cur(Stat, idx, idx_tlt, name, T)
    mean_SPKM.append(Mean)

np.save("mean_SPK.npy", mean_SPKM)

# Plotting of Exc. and Inh. mean activity
pop_act(M_ex, M_in, I_E_v, I_I_v)

