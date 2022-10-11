#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:12:56 2021

@author: sim1bet
"""

import numpy as np
import matplotlib

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

@jit
def in_deg(J,l):
    # Computation of the in-degree measure and related graph plot
    in_deg_dic = nx.in_degree_centrality(J)
    temp = []
    
    for n in range(l):
        temp.append(in_deg_dic[n])
    in_deg_ar = np.array(temp) 
          
    # Saving of in degree scores in .npy    
    np.save("InDegree.npy",in_deg_ar)
    
    return in_deg_ar
    
@jit
def betw(J,l):
    # Computation of betweenness centrality measure and related graph plot
    v_betw = nx.betweenness_centrality(J, weight="weight")
    temp = []
    
    for n in range(l):
        temp.append(v_betw[n])
    betw_vec = np.array(temp)
    
    # Generation of the vector with betw scores and saving in .npy
    np.save("V_Betweenness.npy",betw_vec)
    
    return betw_vec
    
@jit
def clos(J,l):
    # Computation of the closeness centrality measure and related graph plot
    v_clos = nx.closeness_centrality(J)
    temp = []
    
    for n in range(l):
        temp.append(v_clos[n])
    clos_vec = np.array(temp)
    
    # generation of the vector with the closeness centrality score and saving in .npy
    np.save("V_Closeness.npy", clos_vec)
    
    return clos_vec
    
@jit
def katz(J,l):
    # Computation of the eigencentrality measure
    v_kat = nx.katz_centrality_numpy(J, alpha=0.9, weight="weight")
    temp = []
    
    for n in range(l):
        temp.append(v_kat[n])
    kat_vec = np.array(temp)
    
    # Generation of the vector with eig_vec scores and saving in .npy
    np.save("V_Katz.npy", kat_vec)
    
    return kat_vec
    
@jit
def comm(J,l):
    # Computation of the communicability betweenness measure
    v_com = nx.communicability_betweenness_centrality(J)
    temp = [] 
    
    for n in range(l):
        temp.append(v_com[n])
    com_vec = np.array(temp)
    
    # Generation of the vector with com_vec scores and saving in .npy
    np.save("V_Communicability.npy", com_vec)
    
    return com_vec
    
@jit
def curr(J,l):
    # Computation of the current flow centrality measure
    v_cur = nx.current_flow_betweenness_centrality(J,weight="weight",solver="lu")
    temp = []
    
    for n in range(l):
        temp.append(v_cur[n])
    cur_vec = np.array(temp)
    
    # Generation of the vector with cur_vec scores and saving in .npy
    np.save("V_Current.npy", cur_vec)
    
    return cur_vec

@jit
def perc(J,l):
    # Computation of the percolation centrality measure    
    v_perc = nx.percolation_centrality(J, weight="weight")
    temp = []
    
    for n in range(l):
        temp.append(v_perc[n])
    perc_vec = np.array(temp)
    
    # Generation of the vector with perc_vec scores and saving in .npy
    np.save("V_Percolation.npy", perc_vec)
    
    return perc_vec
    
@jit
def pgrnk(J,l):
    # Computation of the pagerank measure for each node
    v_pgr = nx.pagerank_numpy(J, weight="weight")
    temp = []
    
    for n in range(l):
        temp.append(v_pgr[n])
    pgr_vec = np.array(temp)
    
    np.save("V_Pagerank.npy",pgr_vec)
    
    return pgr_vec
    
@jit
def loc_clust(J,l):
    # Computation of the local clustering coefficient for each node
    loc_cl = nx.clustering(J, weight="weight")
    temp = []
    
    for n in range(l):
        temp.append(loc_cl[n])
    loc_cl_vec = np.array(temp)
    
    # Generation of the local clustering vector and saving in .npy
    np.save("LocalClustering.npy", loc_cl_vec)
    
    return loc_cl_vec
    

##############################################################################
# Main for the processing of the graph
##############################################################################
@jit
def graph_an(J_W):
    
    J , l = Build_Graph(J_W)
    
    # Computation of in-degree
    InDeg = in_deg(J,l)
    print('In Deg: Done')
    
    # Computation of centrality measures
    # Betweenness, Eigen, Closeness, Communicability, Current, Percolation
    Betw = betw(J,l)
    print('Betw: Done')
    Clos = clos(J,l)
    print('Clos: Done')
    Katz = katz(J,l)
    print('Kat: Done')
    
    # Computation of local clustering measure 
    LocCl = loc_clust(J,l)
    print('Clust: Done')
    
    # Computation of pagerank
    PgRk = pgrnk(J,l)
    print('Pgr: Done')  
    
    return InDeg, Betw, Clos, Katz, LocCl, PgRk
