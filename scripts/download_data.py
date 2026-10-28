"""Download verified public release shards and extract the completed dataset."""
from pathlib import Path
import hashlib,json,tarfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/microwave173/fixed5-anchor-distill/releases/download/full5000-v1/'
CACHE=ROOT/'downloads';CACHE.mkdir(exist_ok=True)
def download(name):
 p=CACHE/name;tmp=p.with_suffix(p.suffix+'.partial')
 with urllib.request.urlopen(BASE+name,timeout=120) as response,tmp.open('wb') as f:
  while True:
   chunk=response.read(1024*1024)
   if not chunk:break
   f.write(chunk)
 tmp.replace(p);return p
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
manifest=json.loads(download('manifest.json').read_text())
for part in manifest['parts']:
 p=CACHE/part['name']
 if not p.exists() or sha(p)!=part['sha256']:p=download(part['name'])
 assert sha(p)==part['sha256'],f'Checksum mismatch: {p.name}'
 print('Verified; extracting',p.name,flush=True)
 with tarfile.open(p,'r:gz') as archive:
  for member in archive.getmembers():
   target=(ROOT/member.name).resolve()
   assert target.is_relative_to(ROOT/'data/full5000') and member.isfile(),member.name
  archive.extractall(ROOT)
print('Ready:',ROOT/'data/full5000')
