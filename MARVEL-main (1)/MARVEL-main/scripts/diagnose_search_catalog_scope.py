"""Read-only diagnosis of public Search catalog versus actor-visible rows."""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.gppo.event_sources import SearchSlotBuilder,PublicHeatPoint

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    case=ROOT/args.case
    with (case/'gppo'/'mission_frames.jsonl').open() as stream:frame=json.loads(next(stream))
    public=frame['public_heat_positions'];catalog=sorted(map(int,public))
    decisions=json.loads((case/'gppo'/'decisions.json').read_text());first=decisions[0]
    actor_rows=first['task_ids'];visible_heat_ids=[tid-100000 for tid in actor_rows if 100000<=tid<200000]
    # Synthetic equal-priority builder demonstration; no claim that omitted
    # tasks have a certified path or that changing the catalog improves timing.
    points=[PublicHeatPoint(uid,tuple(public[str(uid)])) for uid in catalog]
    builder_rows=SearchSlotBuilder(max_search_uavs=2).build(points)
    source_files=['integrations/gppo/event_sources.py','integrations/gppo/scheduler.py']
    result=dict(public_catalog_heat_ids=catalog,first_actor_heat_ids=visible_heat_ids,
        public_heat_ids_absent_from_first_actor_graph=sorted(set(catalog)-set(visible_heat_ids)),
        first_actor_legal_actions=sum(not value for row in first['mask'] for value in row),
        equal_priority_builder_demo_selected_ids=[row.source_entity_id for row in builder_rows],
        concurrent_search_capacity=2,diagnosis='Candidate catalog is truncated before both GPPO and rule selection.',
        no_simulation_run=True,no_hidden_targets_or_truth_map_read=True,
        omitted_task_feasibility_not_established=True,efficiency_gain_not_established=True,
        native_semantics_must_remain_frozen=True,
        proposed_scope='Extended-only full public pending catalog; separate actual assignment capacity; identical catalog/masks/capacity for GPPO and strong rule.',
        source_hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in source_files})
    with (ROOT/args.out).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
