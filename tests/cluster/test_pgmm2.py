#%%

from graspy.cluster.pgmm import PartitionalGaussianCluster
import numpy as np
import matplotlib.pyplot as plt

pgmm = PartitionalGaussianCluster(max_components=4)

#%%

X = np.random.normal(0, 1, size=(20, 3))
y = pgmm.fit(X)
raise ValueError
# %%

x0 = np.array([[-11.11,-11.09,-10.91,-10.89,-9.11,-9.09,-8.91,-8.89,
                11.11,11.09,10.91,10.89,9.11,9.09,8.91,8.89]]).T

x = np.concatenate((x0,x0),axis=1)
n_samples = x.shape[0]

y = pgmm.fit(x)

print(y)

# %%
f, axes = plt.subplots(1,3,sharey=True)


ed = 2
histories = y[:,0:ed]
unq_histories = np.unique(histories,axis=0)
cs = np.zeros(n_samples)
for point in range(n_samples):
    history = y[point,0:ed]
    for i,u_h in enumerate(unq_histories):
        if (history == u_h).all():
            cs[point] = i
axes[0].scatter(x[:,0],x[:,1],c=cs,cmap='Set1')
axes[0].set_title('First Partition')

ed = 3
histories = y[:,0:ed]
unq_histories = np.unique(histories,axis=0)
cs = np.zeros(n_samples)
for point in range(n_samples):
    history = y[point,0:ed]
    for i,u_h in enumerate(unq_histories):
        if (history == u_h).all():
            cs[point] = i
axes[1].scatter(x[:,0],x[:,1],c=cs,cmap='Set1')
axes[1].set_title('Partition 2')

histories = y
unq_histories = np.unique(histories,axis=0)
cs = np.zeros(n_samples)
for point in range(n_samples):
    history = y[point,:]
    for i,u_h in enumerate(unq_histories):
        if (history == u_h).all():
            cs[point] = i
axes[2].scatter(x[:,0],x[:,1],c=cs,cmap='Set1')
axes[2].set_title('Final Partition')

plt.show()

# %%
