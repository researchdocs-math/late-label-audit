"""Controlled fixed-risk/fixed-backlog simulation for audit-latency sensitivity."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,time,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from experiment import ROOT,replay_one,BUDGETS

def cell(args):
    idx,risk,frac,scenario,reps=args;N=256;K=int(round(N*risk));loss=np.r_[np.ones(K,dtype=int),np.zeros(N-K,dtype=int)]
    y=loss.copy();conf=np.full(N,.8);sums=None;squares=None;failures={};wrongs={};allkeys=None
    fields=['width','failure','correct','wrong','safe','requested','returned','routine_rate']
    for seed in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([420,seed,K,int(frac*100)]))
        out=replay_one(loss,y,conf,scenario,rng,BUDGETS,.05,.15,pending_fraction=frac)
        values=np.array([[x[k] for k in fields] for x in out])
        if sums is None:sums=np.zeros_like(values);squares=np.zeros_like(values);allkeys=[(x['budget'],x['time'],x['method']) for x in out]
        sums+=np.nan_to_num(values);squares+=np.nan_to_num(values)**2
        trial={};bad={}
        for x in out:
            key=(x['budget'],x['method']);trial[key]=max(trial.get(key,0),x['failure']);bad[key]=max(bad.get(key,0),x['wrong'])
        for key,v in trial.items():failures[key]=failures.get(key,0)+v;wrongs[key]=wrongs.get(key,0)+bad[key]
    rows=[]
    for i,key in enumerate(allkeys):
        d={'risk':K/N,'nominal_risk':risk,'pending_fraction':frac,'scenario':scenario,'budget':key[0],'time':key[1],'method':key[2],'replications':reps}
        for j,f in enumerate(fields):
            m=sums[i,j]/reps;v=max(0,(squares[i,j]-reps*m*m)/(reps-1));d[f]=m;d[f+'_se']=(v/reps)**.5
        d['any_failure']=failures[(key[0],key[2])]/reps;d['any_wrong']=wrongs[(key[0],key[2])]/reps;rows.append(d)
    return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=2000);p.add_argument('--workers',type=int,default=10);a=p.parse_args();start=time.perf_counter()
 grid=list(itertools.product([.05,.10,.20,.30],[.25,.50,.75],['independent','slow4','slow8','audit_missing']))
 jobs=[(i,r,f,s,a.reps) for i,(r,f,s) in enumerate(grid)];rows=[]
 with ProcessPoolExecutor(a.workers) as ex:
  for i,out in enumerate(ex.map(cell,jobs)):
   rows.extend(out);print(f'{i+1}/{len(jobs)} cells',flush=True)
 d=pd.DataFrame(rows).sort_values(['nominal_risk','pending_fraction','scenario','budget','time','method']);d.to_csv(ROOT/'results'/'synthetic.csv',index=False,float_format='%.12g')
 (ROOT/'results'/'synthetic_config.json').write_text(json.dumps(vars(a)|{'seconds':time.perf_counter()-start,'N':256,'grid':grid,'alpha':.05,'threshold':.15},indent=2))
if __name__=='__main__':main()
