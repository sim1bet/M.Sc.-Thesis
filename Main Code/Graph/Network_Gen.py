# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:26:17 2021

@author: smnbe
"""


import numpy as np
import networkx as nx
from networkx.generators.random_graphs import *
from numba import jit

@jit
def ErdosRen(w, K, N_E, N_I, J_EE, J_IE, J_EI, J_II):
    if w == "fixed":
        # Constructions of the blocks
        J_EE_m = J_EE*np.random.choice([0,1], (N_E, N_E), p=[1-(K/N_E),K/N_E])/np.sqrt(K)
        J_IE_m = J_IE*np.random.choice([0,1], (N_I, N_E), p=[1-(K/N_I),K/N_I])/np.sqrt(K)
        J_EI_m = J_EI*np.random.choice([0,1], (N_E, N_I), p=[1-(K/N_E),K/N_E])/np.sqrt(K)
        J_II_m = J_II*np.random.choice([0,1], (N_I, N_I), p=[1-(K/N_I),K/N_I])/np.sqrt(K)
        
        # building of the coupling full matrix
        J_W = np.concatenate((np.concatenate((J_EE_m, J_EI_m), axis=1), np.concatenate((J_IE_m, J_II_m), axis=1)), axis = 0)
        rand = False
    else:
        # Constructions of the blocks
        J_EE_W = np.multiply(np.sign(J_EE)*np.random.choice([0,1], (N_E, N_E), p=[1-(K/N_E),K/N_E])/np.sqrt(K),np.random.gamma(5.0,3.0,(N_E,N_E)))
        J_IE_W = np.multiply(np.sign(J_IE)*np.random.choice([0,1], (N_I, N_E), p=[1-(K/N_I),K/N_I])/np.sqrt(K),np.random.gamma(5.0,3.0,(N_I,N_E)))
        J_EI_W = np.multiply(np.sign(J_EI)*np.random.choice([0,1], (N_E, N_I), p=[1-(K/N_E),K/N_E])/np.sqrt(K),np.random.gamma(5.0,3.0,(N_E,N_I)))
        J_II_W = np.multiply(np.sign(J_II)*np.random.choice([0,1], (N_I, N_I), p=[1-(K/N_I),K/N_I])/np.sqrt(K),np.random.gamma(5.0,3.0,(N_I,N_I)))

        # building of the coupling full matrix
        J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)
        rand = False

    return J_W, rand

@jit
def Barabasi(w, K, N_E, N_I, J_EE, J_IE, J_EI, J_II):
    if w == 'fixed':
        # Construction of the Barabasi network (edges)
        J_Gr_Barab_EE = barabasi_albert_graph(n=int(N_E), m=int(K))
        J_Gr_Barab_IE = barabasi_albert_graph(n=int(N_I), m=int(K))
        J_Gr_Barab_EI = barabasi_albert_graph(n=int(N_E), m=int(K))
        J_Gr_Barab_II = barabasi_albert_graph(n=int(N_I), m=int(K))
        J_Barab_EE = nx.to_numpy_matrix(J_Gr_Barab_EE)
        J_Barab_IE = nx.to_numpy_matrix(J_Gr_Barab_IE)
        J_Barab_EI = nx.to_numpy_matrix(J_Gr_Barab_EI)
        J_Barab_II = nx.to_numpy_matrix(J_Gr_Barab_II)
        J_Barab = np.concatenate((np.concatenate((J_Barab_EE, J_Barab_EI), axis=1), np.concatenate((J_Barab_IE, J_Barab_II), axis=1)), axis = 0)

        # Construction of the Weight Matrix
        J_EE_W = J_EE*np.ones((N_E,N_E))/np.sqrt(K)
        J_IE_W = J_IE*np.ones((N_I,N_E))/np.sqrt(K)
        J_EI_W = J_EI*np.ones((N_E,N_I))/np.sqrt(K)
        J_II_W = J_II*np.ones((N_I,N_I))/np.sqrt(K)

        # building of the coupling matrix
        J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

        rand = False

    else:
        # Construction of the Barabasi network (edges)
        J_Gr_Barab_EE = barabasi_albert_graph(n=int(N_E), m=int(K))
        J_Gr_Barab_IE = barabasi_albert_graph(n=int(N_I), m=int(K))
        J_Gr_Barab_EI = barabasi_albert_graph(n=int(N_E), m=int(K))
        J_Gr_Barab_II = barabasi_albert_graph(n=int(N_I), m=int(K))
        J_Barab_EE = nx.to_numpy_matrix(J_Gr_Barab_EE)
        J_Barab_IE = nx.to_numpy_matrix(J_Gr_Barab_IE)
        J_Barab_EI = nx.to_numpy_matrix(J_Gr_Barab_EI)
        J_Barab_II = nx.to_numpy_matrix(J_Gr_Barab_II)
        J_Barab = np.concatenate((np.concatenate((J_Barab_EE, J_Barab_EI), axis=1), np.concatenate((J_Barab_IE, J_Barab_II), axis=1)), axis = 0)

        # Construction of the Weight Matrix
        J_EE_W = np.sign(J_EE)*np.random.gamma(5.0,3.0,(N_E,N_E))/np.sqrt(K)
        J_IE_W = np.sign(J_IE)*np.random.gamma(5.0,3.0,(N_I,N_E))/np.sqrt(K)
        J_EI_W = np.sign(J_EI)*np.random.gamma(5.0,3.0,(N_E,N_I))/np.sqrt(K)
        J_II_W = np.sign(J_II)*np.random.gamma(5.0,3.0,(N_I,N_I))/np.sqrt(K)

        # building of the coupling matrix
        J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

        # If the next lines are uncommented self.rand --> True
        #nod = np.arange(self.N_E+self.N_I)
        #np.random.shuffle(nod)

        #J_W[nod,:]=J_W
        #self.nod = nod
        rand = False

    # Final coupling matrix
    J_W = np.multiply(J_Barab, J_W)

    return J_W, rand

@jit
def WattsStr(w, K, N_E, N_I, J_EE, J_IE, J_EI, J_II):
    if w == 'fixed':
        # Construction of the Barabasi network (edges)
        J_Gr_WS_EE = watts_strogatz_graph(n=int(N_E), k=int(K), p=0.2)
        J_Gr_WS_IE = watts_strogatz_graph(n=int(N_I), k=int(K), p=0.2)
        J_Gr_WS_EI = watts_strogatz_graph(n=int(N_E), k=int(K), p=0.2)
        J_Gr_WS_II = watts_strogatz_graph(n=int(N_I), k=int(K), p=0.2)
        J_WS_EE = nx.to_numpy_matrix(J_Gr_WS_EE)
        J_WS_IE = nx.to_numpy_matrix(J_Gr_WS_IE)
        J_WS_EI = nx.to_numpy_matrix(J_Gr_WS_EI)
        J_WS_II = nx.to_numpy_matrix(J_Gr_WS_II)
        J_WS = np.concatenate((np.concatenate((J_WS_EE, J_WS_EI), axis=1), np.concatenate((J_WS_IE, J_WS_II), axis=1)), axis = 0)

        # Construction of the Weight Matrix
        J_EE_W = J_EE*np.ones((N_E,N_E))/np.sqrt(K)
        J_IE_W = J_IE*np.ones((N_I,N_E))/np.sqrt(K)
        J_EI_W = J_EI*np.ones((N_E,N_I))/np.sqrt(K)
        J_II_W = J_II*np.ones((N_I,N_I))/np.sqrt(K)

        # building of the coupling matrix
        J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

        rand = False

    else:
        # Construction of the Barabasi network (edges)
        J_Gr_WS_EE = watts_strogatz_graph(n=int(N_E), k=int(K), p=0.2)
        J_Gr_WS_IE = watts_strogatz_graph(n=int(N_I), k=int(K), p=0.2)
        J_Gr_WS_EI = watts_strogatz_graph(n=int(N_E), k=int(K), p=0.2)
        J_Gr_WS_II = watts_strogatz_graph(n=int(N_I), k=int(K), p=0.2)
        J_WS_EE = nx.to_numpy_matrix(J_Gr_WS_EE)
        J_WS_IE = nx.to_numpy_matrix(J_Gr_WS_IE)
        J_WS_EI = nx.to_numpy_matrix(J_Gr_WS_EI)
        J_WS_II = nx.to_numpy_matrix(J_Gr_WS_II)
        J_WS = np.concatenate((np.concatenate((J_WS_EE, J_WS_EI), axis=1), np.concatenate((J_WS_IE, J_WS_II), axis=1)), axis = 0)

        # Construction of the Weight Matrix
        J_EE_W = np.sign(J_EE)*np.random.gamma(5.0,3.0,(N_E,N_E))/np.sqrt(K)
        J_IE_W = np.sign(J_IE)*np.random.gamma(5.0,3.0,(N_I,N_E))/np.sqrt(K)
        J_EI_W = np.sign(J_EI)*np.random.gamma(5.0,3.0,(N_E,N_I))/np.sqrt(K)
        J_II_W = np.sign(J_II)*np.random.gamma(5.0,3.0,(N_I,N_I))/np.sqrt(K)

        # building of the coupling matrix
        J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

        # If the next lines are uncommented self.rand --> True
        #nod = np.arange(self.N_E+self.N_I)
        #np.random.shuffle(nod)

        #J_W[nod,:]=J_W
        #self.nod = nod
        rand = False

    # Final coupling matrix
    J_W = np.multiply(J_WS, J_W)

    return J_W, rand
