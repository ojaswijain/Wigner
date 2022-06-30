import matplotlib.pyplot as plt
import numpy as np

def plotter(x,y,label,col,save=True):
    y= [np.log10(abs(e)) for e in y]
    plt.scatter(x,y,c=col)
    plt.xlabel("Maximum m argument")
    plt.ylabel("Log10 of relative error")
    plt.title(label)
    if save:
        plt.savefig(f"plots/{label}.png")
    plt.show()