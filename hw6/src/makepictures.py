import matplotlib
import matplotlib.pyplot as plt
import numpy as np

myrmse = np.array([line.split(",") for line in open("rmse_graph.txt","r").readlines()]).T.tolist()
mynll = np.array([line.split(",") for line in open("nll_graph.txt","r").readlines()]).T.tolist()

index = [float(s) for s in list(mynll[0])]
nll = [float(s) for s in list(mynll[1])]
rmse = [float(s) for s in list(myrmse[1])]

plt.plot(index,nll)
plt.savefig("lossFig")
plt.close()
plt.plot(index,rmse)
plt.savefig("rmseFig")
