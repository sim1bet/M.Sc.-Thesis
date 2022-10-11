# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:09:17 2021

@author: smnbe
"""


import numpy as np
import matplotlib.pyplot as plt

def pop_act(M_ex, M_in, I_E_v, I_I_v):
    x = np.linspace(0,1,1000)
    e = np.amax(I_E_v)
    i = np.amax(I_I_v)
    I_E = I_E_v/e
    I_I = I_I_v/i
    
    # Interpolating excitatory activity
    m_sl_ex, m_in_ex = np.polyfit(I_E,M_ex,1)
    f_m_ex = m_in_ex + m_sl_ex*x
    
    # Interpolating inhibitory activity
    m_sl_in, m_in_in = np.polyfit(I_I,M_in,1)
    f_m_in = m_in_in + m_sl_in*x
    
    plt.figure(figsize=(15,10))
    plt.plot(I_E, M_ex, marker='X', color='green', linestyle='', label='Mean Exc. Network Activity')
    plt.plot(x, f_m_ex, color='orange' ,linestyle='-', label='Regression Line Exc. Activity')
    plt.plot(x, x, linestyle='-.')
    plt.xlabel('I')
    plt.ylabel('m')
    plt.legend()
    plt.savefig('Mean Excitatory Activity Graph')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,10))
    plt.plot(I_I, M_in, marker='X', color='blue', linestyle='', label='Mean Inh. Network Activity')
    plt.plot(x, f_m_in, color='red' ,linestyle='-', label='Regression Line Inh. Activity')
    plt.plot(x, x, linestyle='-.')
    plt.xlabel('I')
    plt.ylabel('m')
    plt.legend()
    plt.savefig('Mean Inhibitory Activity Graph')
    plt.show()
    plt.close()
