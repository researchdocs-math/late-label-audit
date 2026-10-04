"""Verify released input/result checksums and optionally rerun paired replays."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args()
 manifest=json.loads((ROOT/'checksums.json').read_text());bad=[]
 for name,digest in manifest.items():
  path=ROOT/name
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:bad.append(name)
 if bad:raise RuntimeError('Checksum differences: '+', '.join(bad))
 print(f'Verified {len(manifest)} frozen input and numerical result checksums.')
 if a.replay:
  from experiment import task,SCENARIOS,BUDGETS
  reference=pd.read_csv(ROOT/'results/main/seed_metrics.csv.gz');keys=['stream','seed','scenario','budget','time','method']
  for stream in ['gas_lr','gas_rf','hydraulic_lr','hydraulic_rf','gas_mlp']:
   for seed in [0,1]:
    got,_,_,_=task((stream,seed,SCENARIOS,BUDGETS,.05,.15));expected=reference[(reference.stream==stream)&(reference.seed==seed)]
    got=got.sort_values(keys).reset_index(drop=True);expected=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[keys],expected[keys],check_dtype=False)
    cols=[c for c in got if c not in keys]
    np.testing.assert_allclose(got[cols],expected[cols],rtol=1e-10,atol=1e-11,equal_nan=True)
   print(stream,'independent replays match')
if __name__=='__main__':main()
