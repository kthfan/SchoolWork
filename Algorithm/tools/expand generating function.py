import sympy as sp
from sympy.abc import x,y,z,k
import numpy as np

def expand_generating_function(d):
  if isinstance(d, sp.Expr):
    d = sp.Poly((1/f).expand()).all_coeffs()
    d.reverse()
    d = np.array(d, dtype=np.float32)

  r = []
  a = np.array([1], dtype=np.float32)
  for i in range(10):
    op_idx = 0
    s = 0
    for j in range(len(a)):
      if np.abs(a[j]) > 1e-6:
        s = a[j] / d[0]
        op_idx = j
        break
      if j > 0: r.append(0)
    if s == 0: break
    a = a[op_idx:]
    a = np.concatenate([a, np.zeros(len(d) - len(a))])
    # print(a)
    r.append(s)
    a = a - s*d
    # print(a)
    a[np.abs(a)<1e-6] = 0
  return np.array(r, dtype=np.int32)


# f=1/(1-x)*1/(1-x**2)*1/(1-x**5)
# d = np.array([1, -1, -1, 0, 0, -1])

# expand_generating_function(f)
