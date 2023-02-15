import numpy as np
import matplotlib.pyplot as plt
from newv import main
from iso import iso
from eq import eq

if __name__ =="__main__":
    
    size=[]
    time=[]
    gaunts=[]
    l=[5,10,20,50,100,200]
    print("############## Equilateral Bispectrum (l1==l2==l3) ##############")
    for i in l:
        s,t,g = main(i)
        size.append(s)
        time.append(t)
        gaunts.append(g)
    plt.plot(size,time,c='r')
    plt.xlabel("l_max")
    plt.ylabel("Time in Seconds")
    label = "l_max vs Time"
    plt.title(label)
    plt.savefig(f"plots/eq_{label}.png")
    plt.show()

    plt.plot(gaunts,time,c='b')
    plt.xlabel("No. of Gaunts")
    plt.ylabel("Time in Seconds")
    label = "Gaunts vs Time"
    plt.title(label)
    plt.savefig(f"plots/eq_v{label}.png")
    plt.show()

    print("############## Isosceles Bispectrum (l1==l2) ##############")
    for i in l:
        s,t,g = iso(i)
        size.append(s)
        time.append(t)
        gaunts.append(g)
    plt.plot(size,time,c='r')
    plt.xlabel("l_max")
    plt.ylabel("Time in Seconds")
    label = "l_max vs Time"
    plt.title(label)
    plt.savefig(f"plots/iso_{label}.png")
    plt.show()

    plt.plot(gaunts,time,c='b')
    plt.xlabel("No. of Gaunts")
    plt.ylabel("Time in Seconds")
    label = "Gaunts vs Time"
    plt.title(label)
    plt.savefig(f"plots/iso_{label}.png")
    plt.show()


