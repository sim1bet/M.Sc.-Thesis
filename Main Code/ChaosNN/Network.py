# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:26:17 2021

@author: smnbe
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.random_graphs import *
from numba import jit


class ChaosNN:
    # Definition of the chaotic network
    def __init__(self, N_E, N_I, theta_E, theta_I, J_EE, J_EI, J_IE, J_II, T, stoc):
        # Number of Units
        self.N_E = N_E
        self.N_I = N_I
        # Threshold
        self.theta = np.concatenate((theta_E*np.ones((N_E,1)),theta_I*np.ones((N_I,1)))).squeeze()
        # Running time
        self.T = T
        # Weights for the couplings
        self.J_EE = J_EE
        self.J_EI = J_EI
        self.J_IE = J_IE
        self.J_II = J_II
        # Connectivity
        self.K = np.floor(0.1*min(N_E,N_I))
        # Fixed or Stochastic input
        self.stoc = stoc

    @jit
    def J_W_Couplings(self, w):
        if w == "fixed":
            # Constructions of the blocks
            J_EE_m = self.J_EE*np.random.choice([0,1], (self.N_E, self.N_E), p=[1-(self.K/self.N_E),self.K/self.N_E])/np.sqrt(self.K)
            J_IE_m = self.J_IE*np.random.choice([0,1], (self.N_I, self.N_E), p=[1-(self.K/self.N_I),self.K/self.N_I])/np.sqrt(self.K)
            J_EI_m = self.J_EI*np.random.choice([0,1], (self.N_E, self.N_I), p=[1-(self.K/self.N_E),self.K/self.N_E])/np.sqrt(self.K)
            J_II_m = self.J_II*np.random.choice([0,1], (self.N_I, self.N_I), p=[1-(self.K/self.N_I),self.K/self.N_I])/np.sqrt(self.K)
        
            # building of the coupling full matrix
            self.J_W = np.concatenate((np.concatenate((J_EE_m, J_EI_m), axis=1), np.concatenate((J_IE_m, J_II_m), axis=1)), axis = 0)
            self.rand = False
        else:
            # Constructions of the blocks
            J_EE_W = np.multiply(np.sign(self.J_EE)*np.random.choice([0,1], (self.N_E, self.N_E), p=[1-(self.K/self.N_E),self.K/self.N_E])/np.sqrt(self.K),np.random.gamma(5.5,3.0,(self.N_E,self.N_E)))
            J_IE_W = np.multiply(np.sign(self.J_IE)*np.random.choice([0,1], (self.N_I, self.N_E), p=[1-(self.K/self.N_I),self.K/self.N_I])/np.sqrt(self.K),np.random.gamma(5.0,3.0,(self.N_I,self.N_E)))
            J_EI_W = np.multiply(np.sign(self.J_EI)*np.random.choice([0,1], (self.N_E, self.N_I), p=[1-(self.K/self.N_E),self.K/self.N_E])/np.sqrt(self.K),np.random.gamma(6.0,3.2,(self.N_E,self.N_I)))
            J_II_W = np.multiply(np.sign(self.J_II)*np.random.choice([0,1], (self.N_I, self.N_I), p=[1-(self.K/self.N_I),self.K/self.N_I])/np.sqrt(self.K),np.random.gamma(4.9,3.0,(self.N_I,self.N_I)))

            # building of the coupling full matrix
            self.J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)
            self.rand = False

    @jit
    def J_W_Barabasi(self, w):
        if w == 'fixed':
            # Construction of the Barabasi network (edges)
            J_Gr_Barab_EE = barabasi_albert_graph(n=int(self.N_E), m=int(self.K))
            J_Gr_Barab_IE = barabasi_albert_graph(n=int(self.N_I), m=int(self.K))
            J_Gr_Barab_EI = barabasi_albert_graph(n=int(self.N_E), m=int(self.K))
            J_Gr_Barab_II = barabasi_albert_graph(n=int(self.N_I), m=int(self.K))
            J_Barab_EE = nx.to_numpy_matrix(J_Gr_Barab_EE)
            J_Barab_IE = nx.to_numpy_matrix(J_Gr_Barab_IE)
            J_Barab_EI = nx.to_numpy_matrix(J_Gr_Barab_EI)
            J_Barab_II = nx.to_numpy_matrix(J_Gr_Barab_II)
            J_Barab = np.concatenate((np.concatenate((J_Barab_EE, J_Barab_EI), axis=1), np.concatenate((J_Barab_IE, J_Barab_II), axis=1)), axis = 0)

            # Construction of the Weight Matrix
            J_EE_W = self.J_EE*np.ones((self.N_E,self.N_E))/np.sqrt(self.K)
            J_IE_W = self.J_IE*np.ones((self.N_I,self.N_E))/np.sqrt(self.K)
            J_EI_W = self.J_EI*np.ones((self.N_E,self.N_I))/np.sqrt(self.K)
            J_II_W = self.J_II*np.ones((self.N_I,self.N_I))/np.sqrt(self.K)

            # building of the coupling matrix
            J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

            self.rand = False

        else:
            # Construction of the Barabasi network (edges)
            J_Gr_Barab_EE = barabasi_albert_graph(n=int(self.N_E), m=int(self.K))
            J_Gr_Barab_IE = barabasi_albert_graph(n=int(self.N_I), m=int(self.K))
            J_Gr_Barab_EI = barabasi_albert_graph(n=int(self.N_E), m=int(self.K))
            J_Gr_Barab_II = barabasi_albert_graph(n=int(self.N_I), m=int(self.K))
            J_Barab_EE = nx.to_numpy_matrix(J_Gr_Barab_EE)
            J_Barab_IE = nx.to_numpy_matrix(J_Gr_Barab_IE)
            J_Barab_EI = nx.to_numpy_matrix(J_Gr_Barab_EI)
            J_Barab_II = nx.to_numpy_matrix(J_Gr_Barab_II)
            J_Barab = np.concatenate((np.concatenate((J_Barab_EE, J_Barab_EI), axis=1), np.concatenate((J_Barab_IE, J_Barab_II), axis=1)), axis = 0)

            # Construction of the Weight Matrix
            J_EE_W = np.sign(self.J_EE)*np.random.gamma(5.5,3.0,(self.N_E,self.N_E))/np.sqrt(self.K)
            J_IE_W = np.sign(self.J_IE)*np.random.gamma(5.0,3.0,(self.N_I,self.N_E))/np.sqrt(self.K)
            J_EI_W = np.sign(self.J_EI)*np.random.gamma(6.0,3.2,(self.N_E,self.N_I))/np.sqrt(self.K)
            J_II_W = np.sign(self.J_II)*np.random.gamma(4.9,3.0,(self.N_I,self.N_I))/np.sqrt(self.K)

            # building of the coupling matrix
            J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

            # If the next lines are uncommented self.rand --> True
            #nod = np.arange(self.N_E+self.N_I)
            #np.random.shuffle(nod)

            #J_W[nod,:]=J_W
            #self.nod = nod
            self.rand = False

        # Final coupling matrix
        self.J_W = np.multiply(J_Barab, J_W)

    @jit
    def J_W_WattsStr(self, w):
        if w == 'fixed':
            # Construction of the Barabasi network (edges)
            J_Gr_WS_EE = watts_strogatz_graph(n=int(self.N_E), k=int(self.K), p=0)
            J_Gr_WS_IE = watts_strogatz_graph(n=int(self.N_I), k=int(self.K), p=0)
            J_Gr_WS_EI = watts_strogatz_graph(n=int(self.N_E), k=int(self.K), p=0)
            J_Gr_WS_II = watts_strogatz_graph(n=int(self.N_I), k=int(self.K), p=0)
            J_WS_EE = nx.to_numpy_matrix(J_Gr_WS_EE)
            J_WS_IE = nx.to_numpy_matrix(J_Gr_WS_IE)
            J_WS_EI = nx.to_numpy_matrix(J_Gr_WS_EI)
            J_WS_II = nx.to_numpy_matrix(J_Gr_WS_II)
            J_WS = np.concatenate((np.concatenate((J_WS_EE, J_WS_EI), axis=1), np.concatenate((J_WS_IE, J_WS_II), axis=1)), axis = 0)

            # Construction of the Weight Matrix
            J_EE_W = self.J_EE*np.ones((self.N_E,self.N_E))/np.sqrt(self.K)
            J_IE_W = self.J_IE*np.ones((self.N_I,self.N_E))/np.sqrt(self.K)
            J_EI_W = self.J_EI*np.ones((self.N_E,self.N_I))/np.sqrt(self.K)
            J_II_W = self.J_II*np.ones((self.N_I,self.N_I))/np.sqrt(self.K)

            # building of the coupling matrix
            J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

            self.rand = False

        else:
            # Construction of the Barabasi network (edges)
            J_Gr_WS_EE = watts_strogatz_graph(n=int(self.N_E), k=int(self.K), p=0)
            J_Gr_WS_IE = watts_strogatz_graph(n=int(self.N_I), k=int(self.K), p=0)
            J_Gr_WS_EI = watts_strogatz_graph(n=int(self.N_E), k=int(self.K), p=0)
            J_Gr_WS_II = watts_strogatz_graph(n=int(self.N_I), k=int(self.K), p=0)
            J_WS_EE = nx.to_numpy_matrix(J_Gr_WS_EE)
            J_WS_IE = nx.to_numpy_matrix(J_Gr_WS_IE)
            J_WS_EI = nx.to_numpy_matrix(J_Gr_WS_EI)
            J_WS_II = nx.to_numpy_matrix(J_Gr_WS_II)
            J_WS = np.concatenate((np.concatenate((J_WS_EE, J_WS_EI), axis=1), np.concatenate((J_WS_IE, J_WS_II), axis=1)), axis = 0)

            # Construction of the Weight Matrix
            J_EE_W = np.sign(self.J_EE)*np.random.gamma(5.5,3.0,(self.N_E,self.N_E))/np.sqrt(self.K)
            J_IE_W = np.sign(self.J_IE)*np.random.gamma(5.0,3.0,(self.N_I,self.N_E))/np.sqrt(self.K)
            J_EI_W = np.sign(self.J_EI)*np.random.gamma(6.0,3.2,(self.N_E,self.N_I))/np.sqrt(self.K)
            J_II_W = np.sign(self.J_II)*np.random.gamma(4.9,3.0,(self.N_I,self.N_I))/np.sqrt(self.K)

            # building of the coupling matrix
            J_W = np.concatenate((np.concatenate((J_EE_W, J_EI_W), axis=1), np.concatenate((J_IE_W, J_II_W), axis=1)), axis = 0)

            # If the next lines are uncommented self.rand --> True
            #nod = np.arange(self.N_E+self.N_I)
            #np.random.shuffle(nod)

            #J_W[nod,:]=J_W
            #self.nod = nod
            self.rand = False

        # Final coupling matrix
        self.J_W = np.multiply(J_WS, J_W)

    @jit
    def Current(self, i_e, i_i):
        # Definition of the current vectors
        I_E = i_e*np.ones((self.N_E,1))
        I_I = i_i*np.ones((self.N_I,1))
        
        self.I = np.concatenate((I_E,I_I)).squeeze()

    @jit
    def Prop_act_w(self, iterat):
        # Definition of the initial network state (quiescent)
        new_state = np.zeros((self.N_E+self.N_I,1)).squeeze()
        States = np.zeros((self.N_E+self.N_I,self.T+2))
        States[:,0] = new_state

        Curr_temp = np.zeros((self.N_E+self.N_I,self.T))

        for t in range(self.T):
          if self.stoc == True:
            I_st = np.random.normal(0,0.02*np.mean(self.I),(self.N_E+self.N_I,))
            I_F = self.I + I_st
          else:
            I_F = self.I
          for ne_idx in iterat:
            syn_in = np.dot(self.J_W[ne_idx,:],new_state) + I_F[ne_idx] - self.theta[ne_idx]
            Curr_temp[ne_idx,t] = syn_in
           
            if syn_in > 0: 
               new_state[ne_idx] = 1
            else:
               new_state[ne_idx] = 0

          States[:,t+1] = new_state.squeeze()

        Pos = np.zeros(States.shape)
        for i in range(self.T+1):
          Pos[:,i+1][States[:,i+1]!=0]=i+1
          
        self.Pos = Pos
        self.Curr = Curr_temp
        if self.rand == False:
            self.m_ex = np.mean(States[:self.N_E,:])
            self.m_in = np.mean(States[self.N_E:,:])
        else:
            self.m_ex = np.mean(States[self.nod[:self.N_E],:])
            self.m_in = np.mean(States[self.nod[self.N_E:],:])

    @jit
    def Select_sub(self):
        Shf_Pos = np.copy(self.Pos)
        np.random.shuffle(Shf_Pos)
        Pos_sub = Shf_Pos[:600,:min(np.shape(self.Pos)[1],2001)]
        
        self.Pos_sub = Pos_sub
        
    def plot_raster(self, i_e, i_i):
        
        i = '('+str(round(i_e,2))+','+str(round(i_i,2))+')'
        name = '('+str(i_e)+','+str(i_i)+')'
        name = name.replace('.','_')
        
        txt = 'Raster Plot I = '+i
        txt_tlt = 'Raster Plot (Gamma) I = '+name
        plt.figure(figsize=(15,10))
        plt.eventplot(self.Pos_sub[:,1:], colors='black')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.title(txt)
        plt.savefig(txt_tlt)
        plt.show()
        plt.close()
