# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:09:17 2021

@author: smnbe
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def g_avg(k, avg_vec, std_vec, cm, xlb, ylb, brev):

    # Update of the font size
    plt.rcParams.update({'font.size': 22})

    lb = "K = 100 : "+brev+" = "+str(round(avg_vec[10],2))
    if brev == "CC":
        title = "K - Clustering Coefficient (average)"
    elif brev == "SPL":
        title = "K - Shortest Path Length (average)"
    
    plt.figure(figsize=(15,10))
    plt.errorbar(k, avg_vec, yerr=std_vec, marker='o', color='green', linestyle='', label=lb)
    plt.scatter(k, avg_vec, s=150, c=cm, cmap = matplotlib.cm.plasma)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_title("Avg. Degree")
    plt.title(title)
    plt.savefig(title)
    plt.show()
    plt.close()
