import os,signal,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folder=root/'artifacts/search_service_scan_pair_60_78901_20261010'
needle=b'--out\0artifacts/search_service_scan_pair_60_78901_20261010\0'
pids=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:cmd=(entry/'cmdline').read_bytes()
    except (OSError,PermissionError):continue
    if b'scripts/benchmark_fair_search.py\0' in cmd and needle in cmd:
        pids.append(int(entry.name))
(folder/'invalidated.json').write_text(json.dumps({'accepted':False,'reason':'Interrupted own bounded pair after independent review proved short-prefix scan phase defect; preserve all partial data.','owned_pids':pids},indent=2))
for pid in pids:os.kill(pid,signal.SIGTERM)
print(pids)
