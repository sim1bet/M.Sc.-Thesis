#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:12:56 2021

@author: sim1bet
"""

import numpy as np

import networkx as nx
from numba import jit

##############################################################################
# Body of functions for the construction of the graph and the computation of
# relevant graph theoretic measures
##############################################################################

@jit
def Build_Graph(J_W):
    # Construction of the graph from the Coupling matrix
    l = np.shape(J_W)[0]
    
    J = nx.DiGraph()
    vlist = range(l)
    J.add_nodes_from(vlist)
    
    for i in range(l):
        for j in range(l):
            if J_W[i,j] != 0:
                J.add_edge(i,j,weight=J_W[i,j])
                            
    return J,l

# Average Clustering Coefficient
@jit
def clust(J):
    # Computation of the average clustering coefficient
    cl_coef = nx.average_clustering(J, weight="weight")
    
    return cl_coef

# Average Shortest path
@jit
def avg_sht_path(J):
    # Computation of the average shortest path
    sh_path = nx.average_shortest_path_length(J)

    return sh_path

# Degree
@jit
def deg(J,l):
    # Computation of the in-degree measure and related graph plot
    deg_dic = nx.degree_centrality(J)
    temp = []

    for n in range(l):
        temp.append(deg_dic[n])
    deg_ar = np.array(temp)

    # Saving of in degree scores in .npy
    np.save("Degree.npy",deg_ar)

    return deg_ar

##############################################################################
# Main for the processing of the graph
##############################################################################
@jit
def graph_an(J_W):
    
    J , l = Build_Graph(J_W)
    
    # Average clustering coefficient
    Clust_Coef = clust(J)
    # Average Shortest Path length
    Avg_SPL = avg_sht_path(J)

    # Degree centrality
    Deg_C = deg(J,l)
    # Degree
    Deg = J_W.shape[0]*Deg_C
    # Average Degree
    Avg_Deg = np.mean(Deg)
    
    return Clust_Coef, Avg_SPL, Avg_Deg
