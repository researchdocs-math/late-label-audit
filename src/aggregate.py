"""Recompute numerical research summaries from seed-level experiment outputs."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def aggregate():
 d=pd.read_csv(ROOT/'results/main/seed_metrics.csv.gz');f=pd.read_csv(ROOT/'results/main/simultaneous.csv');s=pd.read_csv(ROOT/'results/synthetic.csv')
 primary=d[(d.budget==64)&(d.time==4)&(d.scenario=='slow4')&~d.stream.eq('gas_mlp')]
 q=primary.pivot(index=['stream','seed'],columns='method',values='correct').groupby('seed').mean();delta=q['Panel envelope']-q['Wait panel'];half=1.96*delta.std(ddof=1)/np.sqrt(len(delta))
 metrics=primary.groupby('method')[['correct','width','wrong']].mean()
 out={'primary':metrics.to_dict('index'),'correct_decision_gain':float(delta.mean()),'gain_mc95':[float(delta.mean()-half),float(delta.mean()+half)],'maximum_envelope_run_failure':float(f[f.method.eq('Panel envelope')].groupby(['stream','scenario','budget']).any_failure.mean().max()),'maximum_synthetic_envelope_failure':float(s[s.method.eq('Panel envelope')].any_failure.max())}
 (ROOT/'results/numerical_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=='__main__':aggregate()
