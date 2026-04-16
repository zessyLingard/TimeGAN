import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

real = pd.read_csv("real_ipds_doH.csv", skiprows=1, header=None).iloc[:,0].dropna().values
gen  = pd.read_csv("covert_ipds.csv",   skiprows=1, header=None).iloc[:,0].dropna().values

# ===== KS TEST =====
stat, pval = ks_2samp(real, gen)

print("KS statistic:", stat)
print("p-value:", pval)

# ===== CDF =====
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

x1, y1 = ecdf(real)
x2, y2 = ecdf(gen)

plt.figure()
plt.plot(x1, y1, label="Real")
plt.plot(x2, y2, label="Generated")
plt.legend()
plt.title("CDF Comparison")
plt.xlabel("IPD (seconds)")
plt.ylabel("CDF")
plt.grid()
plt.show()
