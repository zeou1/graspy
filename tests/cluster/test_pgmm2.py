#%%

from graspy.cluster.pgmm import PartitionalGaussianCluster
import numpy as np
import matplotlib.pyplot as plt

pgmm = PartitionalGaussianCluster(max_components=2)

# %%

x0 = np.random.multivariate_normal([0,0],np.eye(2),size=[10])
x1 = np.random.multivariate_normal([5,5],np.eye(2),size=[10])
x = np.concatenate((x0,x1),axis=0)

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