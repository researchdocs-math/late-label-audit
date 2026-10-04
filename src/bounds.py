"""Exact finite-population intervals and completion-safe audit envelopes."""
from functools import lru_cache
import numpy as np
from scipy.stats import hypergeom, beta

@lru_cache(maxsize=40000)
def hyp_table(M,b,alpha):
    """Two-sided exact hypergeometric inversion for all x=0,...,b.

    Feasible totals k satisfy P_k(X>=x)>alpha/2 and
    P_k(X<=x)>alpha/2. Boundary equality is rejected conservatively.
    """
    M=int(M);b=int(b)
    if not 0<=b<=M: raise ValueError('Require 0 <= b <= M')
    x=np.arange(b+1,dtype=int)
    if b==0:return np.array([0]),np.array([M])
    if b==M:return x.copy(),x.copy()
    lo=x.copy();hi=M-b+x
    while np.any(lo<hi):
        mid=(lo+hi)//2
        good=hypergeom.sf(x-1,M,mid,b)>alpha/2
        hi=np.where(good,mid,hi);lo=np.where(good,lo,mid+1)
    lower=lo.copy();lo=x.copy();hi=M-b+x
    while np.any(lo<hi):
        mid=(lo+hi+1)//2
        good=hypergeom.cdf(x,M,mid,b)>alpha/2
        lo=np.where(good,mid,lo);hi=np.where(good,hi,mid-1)
    return lower,lo.copy()

def hyp(M,b,x,alpha):
    l,u=hyp_table(int(M),int(b),float(alpha));return int(l[x]),int(u[x])

def envelope(N,known_errors,M,b,returned_errors,unreturned,alpha):
    l,u=hyp_table(int(M),int(b),float(alpha))
    return (known_errors+l[returned_errors])/N,(known_errors+u[returned_errors+unreturned])/N

@lru_cache(maxsize=64)
def binomial_table(N,alpha):
    out={}
    for n in range(N+1):
        x=np.arange(n+1)
        if n==0: out[n]=(np.array([0.]),np.array([1.]));continue
        l=np.where(x==0,0,beta.ppf(alpha/2,x,n-x+1))
        u=np.where(x==n,1,beta.ppf(1-alpha/2,x+1,n-x))
        out[n]=(l,u)
    return out

def deterministic(loss,observed):
    N=len(loss);s=int(loss[observed].sum());u=N-int(observed.sum())
    return s/N,(s+u)/N

def intersect(a,b):
    lo=max(a[0],b[0]);hi=min(a[1],b[1])
    # Empty confidence sets can occur on the failure event; abstain safely.
    return (0.,1.) if lo>hi+1e-12 else (lo,hi)
