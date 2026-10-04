"""Independent enumeration checks for the exact audit confidence contract."""
import sys, math, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from bounds import hyp_table,envelope,hyp

class BoundTests(unittest.TestCase):
 def test_exact_coverage_by_combinatorial_enumeration(self):
  for M in range(1,25):
   for b in range(M+1):
    l,u=hyp_table(M,b,.05)
    for K in range(M+1):
     failure=0.
     for x in range(max(0,b-(M-K)),min(b,K)+1):
      p=math.comb(K,x)*math.comb(M-K,b-x)/math.comb(M,b)
      if K<l[x] or K>u[x]:failure+=p
     self.assertLessEqual(failure,.05+1e-12)
 def test_every_censoring_pattern_encloses_complete_interval(self):
  for M in range(1,25):
   for b in range(M+1):
    l,u=hyp_table(M,b,.05)
    self.assertTrue(np.all(np.diff(l)>=0));self.assertTrue(np.all(np.diff(u)>=0))
    for complete_errors in range(b+1):
     for returned_errors in range(complete_errors+1):
      for returned_correct in range(b-complete_errors+1):
       missing=b-returned_errors-returned_correct
       low,high=envelope(M,0,M,b,returned_errors,missing,.05)
       self.assertLessEqual(low,l[complete_errors]/M)
       self.assertGreaterEqual(high,u[complete_errors]/M)
 def test_census_and_no_audits(self):
  self.assertEqual(hyp(10,10,3,.05),(3,3));self.assertEqual(hyp(10,0,0,.05),(0,10))
 def test_fast_correct_returns_can_mislead(self):
  # A population of 256 with 64 errors; a possible panel has 16 errors.
  # The 48 correct audits finish first, while the 16 errors remain pending.
  naive=hyp(256,48,0,.05)[1]/256
  safe=envelope(256,0,256,64,0,16,.05)[1]
  self.assertLess(naive,.25);self.assertGreaterEqual(safe,.25)
if __name__=='__main__':unittest.main()
