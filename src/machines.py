import numpy as np
from scipy.stats import norm, triang, beta, uniform
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Slot machine simulator for a casino with 15 machines

def casino(machine):

    if(machine == 1):
        return round(norm.rvs(loc=0.25,scale=1),2)

    if(machine == 2):
        return round(uniform.rvs(loc=-0.75,scale=1.0),2)

    if(machine==3):
        return round(triang.rvs(loc=-0.25,scale=1.0,c=0.4),2)

    if(machine >=4 and machine <=10):
        upper_lim = 1.5 + 1.5*np.sin(machine)
        return round(beta.rvs(a=1.8,b=2.3,loc=-0.5,scale=upper_lim),2)

    if(machine == 11):
        return round(triang.rvs(loc=-0.5,scale=3.0,c=0.0625),2)

    if(machine >= 12 and machine <= 15):
        upper_lim = 1 + 0.25*np.cos(machine)
        return round(uniform.rvs(loc=-0.4,scale=upper_lim),2)

    if(machine >15):
        print("There are only 15 slot machines!")
        return None
def plot_distribution():

    pos = []
    for i in range(15):
        pos.append(i+1)
    def sample_casino(i):
        sample = []
        for j in range(1000):
            sample.append(casino(i+1))
        return sample

    data = [sample_casino(i) for i in range(15)]

    plt.figure(figsize=(20,8),dpi=300)
    plt.violinplot(data, pos, points=200,vert=True, widths=1.1,
                     showmeans=True, showextrema=True, showmedians=False)
    plt.xlim(0, 17)
    plt.ylim(-4, 4)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='both',nbins=21))
    plt.xlabel('Machine Number',fontsize=14)
    plt.tick_params(axis='x', colors='navy')
    plt.tick_params(axis='y', colors='navy')
    plt.ylabel('Distribution of Rewards',fontsize=14)
    plt.grid(which='major', color ='grey', linestyle='-', alpha=0.8)
    plt.grid(which='minor', color ='grey', linestyle='--', alpha=0.2)
    plt.minorticks_on()
    plt.show()
    print("From the violin plot, the lucky machine is 8")