"""Paired finite-cohort monitoring replay with simulated verification delays."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,platform,hashlib
import numpy as np
import pandas as pd
from bounds import hyp,envelope,deterministic,intersect,binomial_table
ROOT=Path(__file__).resolve().parents[1]
TIMES=np.array([0.,1.,2.,4.,8.,16.,32.])
BUDGETS=[16,32,64,128]
SCENARIOS=['independent','slow4','slow8','fast','class','confidence','natural_missing','audit_missing']
METHODS=['Routine-only','Arrival plug-in','No audit','Known outcomes','Wait panel','Complete prefix','Panel envelope']

def cohorts(df,size):
    out=[]
    for g,d in df.groupby('group',sort=True):
        ids=np.arange(len(d));parts=[ids[k:k+size] for k in range(0,len(d),size)]
        if len(parts)>1 and len(parts[-1])<64:parts[-2]=np.concatenate([parts[-2],parts[-1]]);parts.pop()
        for p in parts:out.append(d.iloc[p].reset_index(drop=True))
    return out

def multiplier(loss,y,conf,scenario):
    if scenario in ['slow4','natural_missing','audit_missing']:return np.where(loss,4.,1.)
    if scenario=='slow8':return np.where(loss,8.,1.)
    if scenario=='fast':return np.where(loss,.25,1.)
    if scenario=='class':return np.where(y==(1 if np.max(y)<=6 else 73),4.,1.)
    if scenario=='confidence':return np.where(conf<.7,4.,1.)
    return np.ones(len(loss))

def replay_one(loss,y,conf,scenario,rng,budgets,alpha,tau,pending_fraction=None):
    N=len(loss);truth=loss.mean();mult=multiplier(loss,y,conf,scenario)
    natural=8*np.exp(.7*rng.standard_normal(N))*mult
    audit=np.exp(.7*rng.standard_normal(N))*mult
    natural_missing=rng.random(N)<.1;audit_missing=rng.random(N)<.1
    if scenario=='natural_missing':natural[natural_missing]=np.inf
    if scenario=='audit_missing':audit[audit_missing]=np.inf
    if pending_fraction is not None:
        weights=np.where(loss,4.,1.);Mfixed=int(round(N*pending_fraction))
        forced=rng.choice(N,Mfixed,replace=False,p=weights/weights.sum())
        keep=np.zeros(N,dtype=bool);keep[forced]=True
        natural=np.where(keep,8+natural,0.)
    initial=natural<=8;pending=np.flatnonzero(~initial);M=len(pending);e=int(loss[initial].sum());order=rng.permutation(pending)
    records=[];bt=binomial_table(N,alpha/len(TIMES))
    for budget in budgets:
        b=min(budget,M);A=order[:b];checkpoints=sorted(set([n for n in [8,16,32,64,128] if n<=b]+[b]));checkpoints=[n for n in checkpoints if n>0]
        # Audit completion is the earlier of routine verification and requested verification.
        when=np.minimum(audit[A],natural[A]-8)
        path=[]
        for t in TIMES:
            natural_seen=natural<=8+t;returned=when<=t;obs=natural_seen.copy();obs[A[returned]]=True
            c=int(returned.sum());x=int(loss[A[returned]].sum());u=b-c
            deter=deterministic(loss,obs);base=deterministic(loss,natural_seen)
            rn=int(natural_seen.sum());rx=int(loss[natural_seen].sum());bl,bu=bt[rn]
            vals={'Routine-only':(float(bl[rx]),float(bu[rx])),'No audit':base,'Known outcomes':deter}
            if M==0:arr=panel=wait=prefix=(truth,truth)
            else:
                l,h=hyp(M,c,x,alpha/len(TIMES));arr=((e+l)/N,(e+h)/N)
                panel=intersect(envelope(N,e,M,b,x,u,alpha),deter)
                wait=intersect(envelope(N,e,M,b,x,0,alpha),deter) if c==b else deter
                prefix=deter
                for n in checkpoints:
                    if np.all(returned[:n]):
                        z=int(loss[A[:n]].sum());l,h=hyp(M,n,z,alpha/max(1,len(checkpoints)))
                        prefix=intersect(prefix,((e+l)/N,(e+h)/N))
            vals.update({'Arrival plug-in':arr,'Wait panel':wait,'Complete prefix':prefix,'Panel envelope':panel})
            for method,(l,h) in vals.items():
                decision=-1 if h<=tau else (1 if l>tau else 0)
                correct=(decision==-1 and truth<=tau) or (decision==1 and truth>tau)
                wrong=(decision==-1 and truth>tau) or (decision==1 and truth<=tau)
                path.append({'budget':budget,'time':t,'method':method,'lower':l,'upper':h,'width':h-l,'failure':int(truth<l-1e-12 or truth>h+1e-12),'decision':decision,'correct':int(correct),'wrong':int(wrong),'safe':int(decision==-1),'truth':truth,'pending_fraction':M/N,'requested':(0 if method in ['No audit','Routine-only'] else b),'returned':(0 if method in ['No audit','Routine-only'] else c),'N':N,'routine_rate':rx/rn if rn else np.nan})
        records.extend(path)
    return records

def task(args):
    stream,seed,scenarios,budgets,alpha,tau=args
    df=pd.read_csv(ROOT/'data'/f'{stream}.csv');parts=cohorts(df,256 if stream.startswith('gas') else 128)
    rows=[]
    for scenario_idx,scenario in enumerate(scenarios):
        for j,d in enumerate(parts):
            # Shared random numbers across delay scenarios and all methods.
            rng=np.random.default_rng(np.random.SeedSequence([20261004,seed,j,0 if stream.startswith('gas') else 1]))
            r=replay_one(d.loss.to_numpy(),d.y.to_numpy(),d.confidence.to_numpy(),scenario,rng,budgets,alpha/len(parts),tau)
            for x in r:x.update(stream=stream,seed=seed,scenario=scenario,cohort=j,group=int(d.group.iloc[0]))
            rows.extend(r)
    out=pd.DataFrame(rows)
    # Average within replay seed; retain experiment-level failures for simultaneous coverage.
    keys=['stream','seed','scenario','budget','time','method']
    means=out.groupby(keys,sort=False)[['width','failure','correct','wrong','safe','truth','pending_fraction','requested','returned','routine_rate']].mean().reset_index()
    anyfail=out.groupby(['stream','seed','scenario','budget','method'],sort=False).agg(any_failure=('failure','max'),any_wrong=('wrong','max')).reset_index()
    paths=out if seed==0 else None
    return means,anyfail,paths,len(parts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=200);p.add_argument('--workers',type=int,default=10);p.add_argument('--streams',nargs='+',default=['gas_lr','gas_rf','hydraulic_lr','hydraulic_rf','gas_mlp']);p.add_argument('--scenarios',nargs='+',default=SCENARIOS);p.add_argument('--budgets',nargs='+',type=int,default=BUDGETS);p.add_argument('--output',default='main');p.add_argument('--alpha',type=float,default=.05);p.add_argument('--threshold',type=float,default=.15);a=p.parse_args()
    dest=ROOT/'results'/a.output;dest.mkdir(exist_ok=True);start=time.perf_counter();ms=[];fs=[];ps=[];js={}
    jobs=[(stream,seed,a.scenarios,a.budgets,a.alpha,a.threshold) for stream in a.streams for seed in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(task,j):j for j in jobs}
        for k,f in enumerate(as_completed(futs)):
            m,fail,path,J=f.result();ms.append(m);fs.append(fail);js[futs[f][0]]=J
            if path is not None:ps.append(path)
            if (k+1)%25==0:print(f'{k+1}/{len(jobs)} replays, {time.perf_counter()-start:.1f}s',flush=True)
    metrics=pd.concat(ms).sort_values(['stream','seed','scenario','budget','time','method']);fail=pd.concat(fs).sort_values(['stream','seed','scenario','budget','method'])
    metrics.to_csv(dest/'seed_metrics.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0});fail.to_csv(dest/'simultaneous.csv',index=False)
    pd.concat(ps).sort_values(['stream','scenario','cohort','budget','time','method']).to_csv(dest/'example_paths.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    summary=metrics.groupby(['stream','scenario','budget','time','method']).agg({c:['mean','std'] for c in ['width','failure','correct','wrong','safe','pending_fraction','requested','returned','truth','routine_rate']});summary.to_csv(dest/'summary.csv')
    config=vars(a)|{'seconds':time.perf_counter()-start,'cohorts':js,'times':TIMES.tolist(),'platform':platform.platform(),'prediction_sha256':{s:hashlib.sha256((ROOT/'data'/f'{s}.csv').read_bytes()).hexdigest() for s in a.streams}}
    (dest/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config,indent=2))
if __name__=='__main__':main()
