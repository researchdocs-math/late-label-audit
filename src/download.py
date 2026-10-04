"""Download and verify the original public sensor archives."""
from pathlib import Path
import hashlib,requests
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'gas.zip':('https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip','91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4'),'hydraulic.zip':('https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip','24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2')}
if __name__=='__main__':
 dest=ROOT/'data'/'raw';dest.mkdir(exist_ok=True)
 for name,(url,expected) in SOURCES.items():
  p=dest/name
  if not p.exists():
   r=requests.get(url,timeout=180);r.raise_for_status();p.write_bytes(r.content)
  actual=hashlib.sha256(p.read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError(f'Checksum mismatch for {name}: {actual}')
  print(name,actual)
