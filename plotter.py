import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.special import softmax


def plotter(x,y,label,col,save=True):
    y= [(np.log10(abs(e)+1e-45)) for e in y]
    plt.scatter(x,y,c=col)
    plt.xlabel("Max m argument")
    plt.ylabel("Log10 of relative error")
    plt.title(label)
    if save:
        plt.savefig(f"plots/{label}.png")
    plt.show()

def plot_time(x,y,label,col,save=True):
    plt.plot(x,y,c=col)
    plt.xlabel("No. of Wigners")
    plt.ylabel("Time in Seconds")
    plt.title(label)
    if save:
        plt.savefig(f"plots/{label}.png")
    plt.show()

def plot_wig(x,y,label,col,save=True):
    plt.plot(x,y,c=col)
    plt.xlabel("l Value")
    plt.ylabel("Time in Seconds")
    plt.title(label)
    if save:
        plt.savefig(f"plots/{label}.png")
    plt.show()

def heatmap(x,y,z,label,save=True):
    z= [(0.1*(np.log10(abs(e)+1e-15))+15) for e in z]
    # z= [(1+np.sign(e)) for e in z]
    x=np.array(x)
    y=np.array(y)
    # a=-(x+y)
    color= ['red' if l == 0 else 'blue' if l == 1 else 'green' for l in z]
    # fig = plt.figure()
    # z=softmax(z)
    data = pd.DataFrame(data={'x':x, 'y':y, 'z':z})
    data = data.pivot(index='x', columns='y', values='z')
    # data = gaussian_filter(data, sigma=1)
    sns.heatmap(data, cmap="YlGnBu")
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(x, y, a, color=color)
    if save:
        plt.savefig(f"plots/{label}.png")
    plt.show()