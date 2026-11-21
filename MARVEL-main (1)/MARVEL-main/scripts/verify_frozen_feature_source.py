"""Verify feature/model contract against recorded freeze Git commit, read-only."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('/home/nick/MARVEL')
ref=(FROZEN/'artifacts/phase14_v9_frozen/source_commit.txt').read_text().strip()
patch=(FROZEN/'artifacts/phase14_v9_frozen/uncommitted_changes.patch').read_text()
results=[]
for name in ('task_graph.py','gppo_model.py'):
    assert f'diff --git a/utils/{name} ' not in patch
    old=subprocess.check_output(['git','-C',str(FROZEN),'show',f'{ref}:utils/{name}'])
    now=(ROOT/'integrations/gppo'/name).read_bytes()
    normalized=old.replace(b'from utils.task_graph import',b'from .task_graph import')
    assert now==normalized,name
    results.append({'file':name,'freeze_commit_sha256':hashlib.sha256(old).hexdigest(),
        'current_sha256':hashlib.sha256(now).hexdigest(),'identical_after_import_relocation':True,
        'modified_by_freeze_patch':False})
result={'freeze_commit':ref,'freeze_patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),
    'freeze_environment_agents':4,'freeze_document':'/home/nick/MARVEL/artifacts/phase14_v9_frozen/FREEZE.md',
    'checkpoint':json.loads((ROOT/'artifacts/search_frozen_training_metadata_20261010.json').read_text())['checkpoint_sha256'],
    'files':results,'production_changed':False,'training':False,
    'limitation':'Freeze document declares four-agent environment; checkpoint does not store n_agents. Source identity excludes a normalization/model-copy mismatch but does not prove every runtime observation is correct or policy generalization impossible.'}
out=ROOT/'artifacts/search_frozen_feature_contract_commit_verified_20261010.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
