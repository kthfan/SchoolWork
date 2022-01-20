import numpy as np
import matplotlib.pyplot as plt


def show_time_complexity(n, expr, theta=None, c=None):
  
    def T(nn):
      ni = nn>1
      n = nn[ni]
      if len(n) == 0: return np.ones(nn.shape)
      T1 = np.empty(nn.shape)
      T1[~ni] = 1
      T1[ni] = expr(T, n)
      return T1
    x = n
    y = T(x)

    plt.plot(x, y, label="true")

    if theta:
      if c is None: c = 10**np.linspace(-1, 1, 5)
      for cc in c: 
        y = cc*theta(x)
        plt.plot(x, y, label="c={}".format(np.round(cc, 2)))
      plt.legend()

# n = np.linspace(1, 10, 10)
# expr = lambda T, n: 2*T(n/2) + n/np.log(n)**2
# theta = lambda n: n
# show_time_complexity(n , expr, theta)
