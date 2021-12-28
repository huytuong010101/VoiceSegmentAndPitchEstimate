import numpy as np
from matplotlib import pyplot as plt

tmp = np.load("v_u_distribution.npy", allow_pickle=True)
u, v = tmp[0], tmp[1]
plt.clf()
plt.hist(v, bins=np.arange(0, 0.01, 0.002), alpha=0.5, label=["voice STE"])
plt.hist(u[:200], bins=np.arange(0, 0.01, 0.002), alpha=0.5, label=["silent STE"])
plt.legend(prop={'size': 10})
plt.xlabel("Amplitude")
plt.ylabel("Number of smaple")
plt.title("Distribution of STE")
plt.show()


threshold = 0.002
TRUE = FALSE = 0
for item in u:
    if item < threshold:
        TRUE += 1
    else:
        FALSE += 1
for item in v:
    if item >= threshold:
        TRUE += 1
    else:
        FALSE += 1
print("Accuracy = ", TRUE / (TRUE + FALSE))
