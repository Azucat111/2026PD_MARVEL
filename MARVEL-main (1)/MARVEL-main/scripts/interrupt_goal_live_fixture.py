from pathlib import Path
import os,signal,json
name=b"scripts/verify_goal_certificate_live_observed.py"
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except (OSError,PermissionError):continue
 if name in args and any(b'python' in a for a in args[:1]):
  os.kill(int(p.name),signal.SIGINT);found.append(int(p.name))
print(json.dumps(dict(interrupted_owned_fixture=found,reason='Confirmed live base exploration is overwritten by goal-neighbor selection; preserve cycle evidence before minimal recovery ordering repair.')))
