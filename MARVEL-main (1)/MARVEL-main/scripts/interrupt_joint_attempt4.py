"""Interrupt exactly this owned experimental trainer; preserve all files."""
import os,signal,json
from pathlib import Path
expected='artifacts/extended_gppo_joint_batch_20261010_attempt4'
owned=[]
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:args=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
 except (OSError,UnicodeError):continue
 if not any(Path(a).name=='train_extended_gppo_batched.py' for a in args) or '--out' not in args:continue
 if args[args.index('--out')+1]==expected:owned.append(int(entry.name))
assert len(owned)==1,owned
os.kill(owned[0],signal.SIGINT)
print(json.dumps(dict(owned_pid=owned[0],signal='SIGINT',reason='Observed Relay21 cycles despite existing certified observed Dijkstra route; repair execution before collecting more policy data.')))
