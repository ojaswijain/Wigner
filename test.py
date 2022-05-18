import pipeline

l1=1
l2=2
l3=3

m1=1
m2=2
m3=3

idx1=str(l1)+"."+str(l2)+"."+str(l3)+"."
idx2=str(m1)+"."+str(m2)+"."+str(m3)

pipeline.store_val_ana(1,2,3,1,2,3,10)
x=pipeline.give_val_ana(1,2,3,1,2,3)
print(x)
pipeline.store_val_ana(1,2,3,1,2,3,20)
x=pipeline.give_val_ana(1,2,3,1,2,4)
print(x)
x=pipeline.give_val_ana(1,2,3,1,2,3)
print(x)
