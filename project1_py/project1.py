#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_best = nesterov(f, g, x0, n, count, .4, .6)

    return x_best

def nesterov(f, g, x0, n, count, alpha, beta):
    x = x0
    v = np.zeros(len(x0))
    while count() < n:

        g_nest = g(x + beta*v)
        d_nest = -1/np.linalg.norm(g_nest) * g_nest
        v_new = beta*v + alpha*d_nest

        x += v_new
        v = v_new
        alpha *= .8
        

    x_best = x
    return x_best

