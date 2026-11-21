"""Fair pooled Search metric audit. Passing never means whole project completion."""
from pathlib import Path
import argparse,json,math,hashlib
FINAL_SEEDS={101101,202202,303303}

def aggregate_cases(rows):
 if not rows:raise ValueError('No paired cases')
 if len({r['checkpoint_sha256'] for r in rows})!=1:raise ValueError('Cannot pool different models')
 revisions={r.get('source_revision') for r in rows}
 if len(revisions)>1:raise ValueError('Cannot pool different production source revisions')
 keys=[(r['seed'],r['uavs'],r['domain']) for r in rows]
 if len(keys)!=len(set(keys)):raise ValueError('Duplicate cases')
 for r in rows:
  if r['targets']<=0 or not math.isfinite(r['gppo_total']) or not math.isfinite(r['rule_total']) or r['gppo_total']<=0 or r['rule_total']<0:raise ValueError('Invalid latency totals')
 def gain(rs):return sum(r['rule_total'] for r in rs)/sum(r['gppo_total'] for r in rs)-1
 groups={f'{n}_{d}':gain([r for r in rows if r['uavs']==n and r['domain']==d]) for n,d in sorted({(r['uavs'],r['domain']) for r in rows})}
 required={(s,n,d) for s in FINAL_SEEDS for n in (30,60) for d in ('static','dynamic')}
 eligible=set(keys)==required and bool(rows[0].get('source_revision')) and all(r.get('final_eligible',False) and r['horizon_seconds']>=300 and r['valid'] and r['collisions']==0 and r['found']==r['targets'] for r in rows)
 return dict(cases=len(rows),efficiency_gain=gain(rows),group_efficiency_gains=groups,final_search_metric_acceptance=eligible and gain(rows)>=.15,project_goal_complete=False,scope='Search metric only; full Relay/Safety/scalability and native compatibility need separate evidence')

def shared_source_fingerprint(manifest):
 sources=manifest.get('source_hashes',{})
 input_sources={'scripts/extended_gppo_heading.py','scripts/gppo_benchmark_domains.py','scripts/benchmark_fair_search.py','scripts/gppo_acceptance_protocol.py'}
 shared={rel:digest for rel,digest in sources.items()
         if rel.startswith(('utils/','integrations/','configs/','load_model/')) or rel in ('parameter.py','test_parameter.py') or rel in input_sources}
 variant=manifest.get('controller_variant')
 if variant:
  shared['__controller_variant__']=variant
  shared.update({rel:digest for rel,digest in sources.items() if rel.startswith('scripts/')})
 return hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest() if shared else None

def validate_pair_warning_units(manifest,config):
 declared=manifest.get('warning_rate_mps')
 units=config.get('_benchmark_warning_units',{})
 if declared is None:declared=units.get('physical_rate_mps')
 hazards=[o for o in config.get('dynamic_obstacles',[]) if o.get('id')=='natural_expanding_warning']
 if declared is None:
  if hazards and manifest.get('stage')=='heldout final Search acceptance':raise ValueError('Final dynamic case lacks a physical speed declaration')
  return None
 if not math.isfinite(float(declared)) or float(declared)<=0 or len(hazards)!=1:raise ValueError('Invalid physical warning declaration')
 dt=float(config['scenario']['duration']['step_dt'])
 if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid physical warning timestep')
 actual=float(hazards[0]['params']['expansion_rate'])/dt
 if not math.isclose(actual,float(declared),rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Actual physical warning speed disagrees with manifest')
 if units and not math.isclose(float(units['physical_rate_mps']),actual,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Actual physical warning speed disagrees with runtime units')
 return actual

def load_case(folder):
 m=json.loads((folder/'manifest.json').read_text());results=[json.loads((folder/a/'summary.json').read_text()) for a in ('gppo','rule')]
 configs=[json.loads((folder/a/'config.json').read_text()) for a in ('gppo','rule')]
 initial=[json.loads((folder/a/'initial.json').read_text()) for a in ('gppo','rule')]
 if configs[0]!=configs[1] or initial[0]!=initial[1]:raise ValueError('Unequal configs/initial conditions')
 for cfg in configs:validate_pair_warning_units(m,cfg)
 if any(r['candidate_budget']!=0 or not r['natural_from_reset'] or not r['physical_coverage'] for r in results):raise ValueError('Not natural unfiltered physical comparison')
 metrics=[r['metric'] for r in results]
 if metrics[0]['horizon_seconds']!=metrics[1]['horizon_seconds'] or metrics[0]['total_targets']!=metrics[1]['total_targets']:raise ValueError('Unequal horizons/targets')
 for r,metric in zip(results,metrics):
  if r['seed']!=m['seed'] or r['uavs']!=m['uavs'] or len(metric['capped_detection_seconds'])!=metric['total_targets']:raise ValueError('Identity/metric mismatch')
  if not math.isclose(sum(metric['capped_detection_seconds'])/metric['total_targets'],metric['mean_capped_seconds']):raise ValueError('Inconsistent metric')
 revision=shared_source_fingerprint(m)
 return dict(source_revision=revision,seed=m['seed'],uavs=m['uavs'],domain='dynamic' if configs[0].get('dynamic_obstacles') else 'static',checkpoint_sha256=m['checkpoint_sha256'],horizon_seconds=metrics[0]['horizon_seconds'],final_eligible=m.get('stage')=='heldout final Search acceptance',valid=all(r['valid_for_comparison'] and r['required_nonrelay_tasks_complete'] for r in results),collisions=sum(sum(r['collision_types'].values()) for r in results),targets=metrics[0]['total_targets'],found=min(x['detected_targets'] for x in metrics),gppo_total=sum(metrics[0]['capped_detection_seconds']),rule_total=sum(metrics[1]['capped_detection_seconds']))

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
 rows=[load_case(root/f) for f in a.cases];result=aggregate_cases(rows);result['rows']=rows
 with (root/a.out).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
