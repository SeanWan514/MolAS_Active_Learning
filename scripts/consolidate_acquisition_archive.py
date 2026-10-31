#!/usr/bin/env python3
"""Lossless release assembly from original data, with pinned Git-blob checks."""
from pathlib import Path
import subprocess,json,hashlib,zipfile,tarfile,gzip,io,collections,time
import argparse
parser=argparse.ArgumentParser(description="Losslessly package original acquisition CSVs; no experiments.")
parser.add_argument('--original-zip',type=Path,required=True)
parser.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--git-tree',type=Path)
parser.add_argument('--local-archives',type=Path,help='Optional available-diagnostic ZIP directory from package_acquisition_archive.py')
args=parser.parse_args()
dest=args.destination.resolve();out=dest/'results/acquisition';out.mkdir(parents=True,exist_ok=True)
commit='f11931c704650b8ecf5ad86a0ae69d87c82a89ec'
tree=args.git_tree or dest/'provenance/pinned_acquisition_tree.txt.gz'
if list(out.glob('*/*.tar.xz')):
 raise FileExistsError('Canonical archives already exist; use a separate --destination to preserve frozen results.')
(dest/'provenance').mkdir(parents=True,exist_ok=True)
rows=[]
for l in gzip.open(tree,'rt').read().splitlines():
 info,path=l.split('\t',1);mode,typ,oid=info.split()
 if typ=='blob':rows.append({'path':path,'git_blob':oid})
byrun=collections.defaultdict(list)
for r in rows:byrun['/'.join(r['path'].split('/')[:3])].append(r)
original_zip=zipfile.ZipFile(args.original_zip)
blob_paths={r['git_blob']:r['path'] for r in rows}
def get_blob(oid):
 raw=original_zip.read('cost_aware_relaxed_six_methods/'+blob_paths[oid]);assert blobid(raw)==oid;return raw
def blobid(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
manifest=[];archivefiles=[];local_records=[];unmatched=[]
for idx,(run,items) in enumerate(sorted(byrun.items())):
 method,fold,seed=run.split('/');old=(args.local_archives or out)/method/f'{fold}_{seed}.zip';target=out/method/f'{fold}_{seed}.tar.xz';target.parent.mkdir(parents=True,exist_ok=True);contents={};sources={}
 if old.exists():
  with zipfile.ZipFile(old) as z:
   for n in z.namelist():
    # Cumulative logs are losslessly deduplicated below. Full JSON restart state
    # repeats the ledger and is not needed to inspect matrices/observations.
    if n.startswith('queried_cells/cumulative/') or n=='run_state/state.json':continue
    raw=z.read(n);contents[n]=raw;sources[n]='unchanged_local_archive'
 for r in items:
  member='/'.join(r['path'].split('/')[3:])
  if '/cumulative/' in member and not member.endswith('100pct.csv'):continue
  raw=get_blob(r['git_blob'])
  if member in contents and contents[member]!=raw:raise RuntimeError('Local/remote conflict: '+r['path'])
  contents[member]=raw;sources[member]='pinned_original_git_blob'
 fullmember='queried_cells/cumulative/queried_100pct.csv';lines=contents[fullmember].splitlines(keepends=True);n=len(lines)-1
 for r in items:
  member='/'.join(r['path'].split('/')[3:]);entry=dict(r);entry['archive']=str(target.relative_to(dest));entry['member']=member
  if '/cumulative/' in member:
   cp=int(member.rsplit('_',1)[1][:3]);count=(n*cp+99)//100;raw=b''.join(lines[:count+1])
   if blobid(raw)!=r['git_blob']:
    unmatched.append(r['path']);raw=get_blob(r['git_blob']);contents[member]=raw;sources[member]='pinned_original_git_blob';entry['encoding']='stored'
   else:entry.update(member=fullmember,encoding='final_ledger_prefix',data_rows=count)
  else:raw=contents[member];entry['encoding']='stored'
  assert blobid(raw)==r['git_blob'];entry.update(status='verified',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());manifest.append(entry)
 with tarfile.open(target,'w:xz',preset=3) as tf:
  for member,raw in sorted(contents.items()):
   info=tarfile.TarInfo(member);info.size=len(raw);info.mode=0o644;info.mtime=1793404800;tf.addfile(info,io.BytesIO(raw))
   local_records.append({'archive':str(target.relative_to(dest)),'member':member,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':sources[member]})
 archivefiles.append({'path':str(target.relative_to(dest)),'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'members':len(contents)})
 print(f'{idx+1}/90 archived {run}: {target.stat().st_size/1e6:.2f} MB',flush=True)

original_zip.close()
for name,data in [('pinned_acquisition_files.jsonl.gz',manifest),('acquisition_stored_members.jsonl.gz',local_records)]:
 with (dest/'provenance'/name).open('wb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
   for r in data:gz.write((json.dumps(r,sort_keys=True)+'\n').encode())
summary={'source_repository':'git@github.com:SeanWan514/PORT-AS_cost_aware.git','pinned_commit':commit,'recovered_from':'original supplied cost_aware_relaxed_six_methods.zip; every recovered CSV verified against pinned Git tree','original_csv_files_verified':len(manifest),'completed_matrices':9000,'incremental_query_logs':9000,'cumulative_query_logs':9000,'runs':90,'methods':sorted({p.split('/')[0] for p in byrun}),'acquisition_seeds':[0,1,2],'downstream_acquisition_seed':0,'cumulative_prefix_mismatches_fetched_original':unmatched,'archive_bytes':sum(a['bytes'] for a in archivefiles),'archives':archivefiles,'note':'Original CSV bytes checked against Git blob identities. Cumulative checkpoint files stored losslessly as verified prefixes of final ledgers. Original incremental-log defects preserved. Auxiliary local diagnostics may remain incomplete; see local inventory. No experiments or numerical scientific results regenerated.'}
(dest/'provenance/pinned_acquisition_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='archives'},indent=2))
