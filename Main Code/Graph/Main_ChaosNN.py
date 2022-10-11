# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:34:11 2021

@author: smnbe
"""

import numpy as np

from Graph_Analysis import *
from Graph_Plot import *
from Network_Gen import *

# Definition of the parameters
N_E = 1000
N_I = 1000
K = np.arange(start=60,stop=144,step=4)

# For Fixed weights
# J_EI = -2.0; J_II = -1.8
# For Random weights
# J_EI = -1.0; J_II = -1.0
J_EE = 1.0
J_IE = 1.0
J_EI = -2.0
J_II = -1.8

# Condition for the weights w = {fixed,gamma}
w = "fixed"
# Number of realization for the  ensemble average
N_rel = 10
# Topology considered for the analysis
top = "WS"

# Vector for the ensemble average and std of the clustering coefficient
ens_avg_cl = np.zeros(K.shape)
ens_std_cl = np.zeros(K.shape)

# Vector for the ensemble average and std of the average shortest path length
ens_avg_SPL = np.zeros(K.shape)
ens_std_SPL = np.zeros(K.shape)

# Vector for the ensemble average of the average degree
ens_avg_deg = np.zeros(K.shape)

it = 0

for k in K:
    # Vector for the single realizations clustering coefficients
    cl_vec = np.zeros((N_rel,))
    # Vector for the single realizations SPL
    SPL_vec = np.zeros((N_rel,))
    # Vector for the single realizations Degree
    Deg_vec = np.zeros((N_rel,))

    for n in range(N_rel):
        if top == "ER":
            J_W, rand = ErdosRen(w, k, N_E, N_I, J_EE, J_IE, J_EI, J_II)
        elif top == "WS":
            J_W, rand = WattsStr(w, k, N_E, N_I, J_EE, J_IE, J_EI, J_II)

        Clust_Coef, Avg_SPL, Avg_Deg = graph_an(J_W)
        # Storing of the values
        cl_vec[n] = Clust_Coef
        SPL_vec[n] = Avg_SPL
        Deg_vec[n] = Avg_Deg

    # Avearge and Std of the vectors and storing of the values
    ens_avg_cl[it] = np.mean(cl_vec)
    ens_std_cl[it] = np.std(cl_vec)

    ens_avg_SPL[it] = np.mean(SPL_vec)
    ens_std_SPL[it] = np.std(SPL_vec)

    ens_avg_deg[it] = np.mean(Deg_vec)

    it += 1

# Plotting of the results for the given ensemble averages

xcl = "K"
ycl = "Clustering Coefficient"

xSPL = xcl
ySPL = "Shortest Path Length"

brev = "CC"
g_avg(K, ens_avg_cl, ens_std_cl, ens_avg_deg, xcl, ycl, brev)

brev = "SPL"
g_avg(K, ens_avg_SPL, ens_std_SPL, ens_avg_deg, xSPL, ySPL, brev)





