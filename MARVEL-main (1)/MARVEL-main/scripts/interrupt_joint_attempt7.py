from pathlib import Path
import os,signal,json
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if b'scripts/train_extended_gppo_batched.py' in args and b'artifacts/extended_gppo_joint_batch_20261010_attempt7' in args:
  os.kill(int(p.name),signal.SIGINT);found.append(int(p.name))
print(json.dumps(dict(interrupted_owned_training=found,reason='Search service-radius semantic regression confirmed at public border heats; retain1 full and current partial episode, zero optimizer updates.')))
