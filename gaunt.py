import numpy as np
import pipeline
import pickle as pkl
import utils

zero=np.zeros(3)


def coeff(l):
    return np.sqrt((2*l[0]+1)*(2*l[1]+1)*(2*l[2]+1)/(4*np.pi))

#l is the list of l1,l2,l3
def gaunt(l):
    ans=0
    l=np.array(l)
    file = open("ana.pkl", "rb")
    pipeline.wigner_dict_ana = pkl.load(file)
    prefactor=coeff(l)*pipeline.give_val_ana(l,zero)
    for m1 in range(-l[0], l[0]+1):
        for m2 in range(-l[1], l[1]+1):
            for m3 in range(-l[2], l[2]+1):
                if m1+m2+m3==0:
                    val=prefactor*pipeline.give_val_ana(l, np.array([m1,m2,m3]))
                    key=utils.to_key(l[0],l[1],l[2],m1,m2,m3)
                    pipeline.gaunt_dic[key]=val
                    ans += 1

    
    with open('gaunt.pkl', 'wb') as handle:
        pkl.dump(pipeline.gaunt_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return ans

if __name__ == "__main__":
    for l in [200]:
        print(gaunt([l,l+1,l+3]))    
    print(pipeline.gaunt_dic)