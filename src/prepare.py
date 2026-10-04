"""Prepare public sensor datasets and frozen out-of-training predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ.setdefault(k,'1')
import argparse, json, hashlib, zipfile, io, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1]

def sensor_stats(args):
    path,name=args
    with zipfile.ZipFile(path) as z: a=np.loadtxt(io.BytesIO(z.read(name)),dtype=np.float32)
    return np.column_stack([a.mean(1),a.std(1),a.min(1),a.max(1)])

def load_data(raw,workers):
    with zipfile.ZipFile(raw/'gas.zip') as z:
        parts=[load_svmlight_file(io.BytesIO(z.read(f'Dataset/batch{i}.dat')),n_features=128) for i in range(1,11)]
    gx=np.vstack([a.toarray() for a,b in parts]);gy=np.concatenate([b for a,b in parts]).astype(int)
    gb=np.concatenate([np.full(len(b),i+1) for i,(a,b) in enumerate(parts)])
    sensors=['PS1','PS2','PS3','PS4','PS5','PS6','EPS1','FS1','FS2','TS1','TS2','TS3','TS4','VS1','CE','CP','SE']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        hx=np.column_stack(list(pool.map(sensor_stats,[(raw/'hydraulic.zip',s+'.txt') for s in sensors])))
    with zipfile.ZipFile(raw/'hydraulic.zip') as z: profile=np.loadtxt(io.BytesIO(z.read('profile.txt')),dtype=int)
    np.savez_compressed(ROOT/'data'/'features.npz',gas_x=gx,gas_y=gy,gas_batch=gb,hydraulic_x=hx,hydraulic_profile=profile)
    return gx,gy,gb,hx,profile

def train_one(args):
    name,modelname,X,y,group,train,workers=args
    t=time.perf_counter()
    if modelname=='lr': model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=3000,random_state=1729))
    else: model=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,max_features='sqrt',random_state=1729,n_jobs=workers)
    model.fit(X[train],y[train]);test=~train;prob=model.predict_proba(X[test]);pred=model.classes_[prob.argmax(1)]
    df=pd.DataFrame({'row_id':np.flatnonzero(test),'group':group[test],'y':y[test],'prediction':pred,'confidence':prob.max(1),'loss':(pred!=y[test]).astype(int)})
    df.to_csv(ROOT/'data'/f'{name}_{modelname}.csv',index=False,float_format='%.12g')
    return {'stream':name+'_'+modelname,'training_n':int(train.sum()),'test_n':int(test.sum()),'features':X.shape[1],'error':float((pred!=y[test]).mean()),'seconds':time.perf_counter()-t,'group_error':df.groupby('group').loss.mean().to_dict()}

def train_mps(gx,gy,gb):
    import torch
    torch.manual_seed(1729);np.random.seed(1729)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    train=gb<=2;sc=StandardScaler().fit(gx[train]);x=sc.transform(gx).astype('float32');classes=np.unique(gy);y=np.searchsorted(classes,gy)
    X=torch.tensor(x[train],device=device);Y=torch.tensor(y[train],device=device)
    model=torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,len(classes))).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.001)
    t=time.perf_counter()
    for epoch in range(400):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(X),Y);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad(): p=model(torch.tensor(x[~train],device=device)).softmax(1).cpu().numpy()
    pred=classes[p.argmax(1)]
    pd.DataFrame({'row_id':np.flatnonzero(~train),'group':gb[~train],'y':gy[~train],'prediction':pred,'confidence':p.max(1),'loss':(pred!=gy[~train]).astype(int)}).to_csv(ROOT/'data'/'gas_mlp.csv',index=False,float_format='%.12g')
    torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'mean':sc.mean_,'scale':sc.scale_,'classes':classes},ROOT/'data'/'gas_mlp_weights.pt')
    return {'stream':'gas_mlp','device':device,'training_n':int(train.sum()),'test_n':int((~train).sum()),'features':128,'error':float((pred!=gy[~train]).mean()),'seconds':time.perf_counter()-t,'epochs':400}

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=ROOT/'data'/'raw');p.add_argument('--workers',type=int,default=10);p.add_argument('--mlp',action='store_true');p.add_argument('--features-only',action='store_true');a=p.parse_args()
    start=time.perf_counter()
    if a.features_only:
        f=np.load(ROOT/'data'/'features.npz');gx,gy,gb,hx,hp=[f[k] for k in ['gas_x','gas_y','gas_batch','hydraulic_x','hydraulic_profile']]
    else:gx,gy,gb,hx,hp=load_data(a.raw,a.workers)
    jobs=[(n,m,X,y,g,tr,max(1,a.workers//4)) for n,X,y,g,tr in [('gas',gx,gy,gb,gb<=2),('hydraulic',hx,hp[:,1],hp[:,0],hp[:,0]==100)] for m in ['lr','rf']]
    with ProcessPoolExecutor(max_workers=4) as e: metrics=list(e.map(train_one,jobs))
    if a.mlp: metrics.append(train_mps(gx,gy,gb))
    out={'models':metrics,'preparation_seconds':time.perf_counter()-start,'raw_sha256':{f:hashlib.sha256((a.raw/f).read_bytes()).hexdigest() for f in ['gas.zip','hydraulic.zip'] if (a.raw/f).exists()},'versions':{}}
    import scipy,sklearn
    out['versions']={'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}
    (ROOT/'results'/'training.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
