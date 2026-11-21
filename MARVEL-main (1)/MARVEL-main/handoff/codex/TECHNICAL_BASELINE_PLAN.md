# Same-condition rule scheduling baseline implementation plan
> Use executing-plans inline; user authorized continuous technical work. No PPT/demo/report, training, commit or push.
**Goal:** Make the >=15% original technical goal auditable against a real traditional rule baseline.
**Architecture:** A pure rule decision reads the SAME TaskGraph valid masks and raw observed total travel+processing times. A benchmark-only binding replaces only the inference decision; task lifecycle, public heats, motion, sensing, capacities, shield, resources, stale positions and seeds stay common.
**Spec:** PROJECT_ACCEPTANCE_LEDGER.md, original question.txt/task_plan.md.

## Constraints
- Frozen native and gppo-integration untouched.
- Never use hidden targets or truth obstacles to choose assignments.
- Priority descending, minimum observed total time, deterministic ties by public task/UAV id.
- Do not use current heuristic hidden-coordinate routing as the fair baseline.
- Predefine primary efficiency before paired experiments: inverse mean capped true-survivor detection time at common H=300 mission seconds; undetected survivors count H. Improvement = rule_mean / GPPO_mean -1. Report completion count, static/dynamic rejected proposals, UAV overlaps, safety/relay lifecycle and runtime separately.
- No positive claim from heat dwell alone, censored tasks, controlled activation or one selected seed.
- Start with a bounded same-condition smoke test, retain failed data. Original500m case is a recovery test; existing80m case can test natural activation only if explicitly labelled its own scenario.
- Dynamic/Relay fixtures must have geometrically and protocol-feasible obligations (comm20m/max2hops), genuine physical traversal and release/preemption events.

## Tasks
1. Create integrations/rule_scheduling.py choose_rule_assignment(graph,deterministic=True); tests priority/cost/masks, capability/busy/alive, context masks, permutation ties and empty graph. RED then GREEN.
2. Benchmark-only allocator binding; audit graph/mask/decision inputs and ensure restoration. Test sequential assignments reserve resources via the existing GPPOEventAllocator.
3. Paired bounded smoke experiment on declared common scenario/seeds; no retuning/cherry-picking after seeing results. Save raw per-target detection times and full scenario/source/map/checkpoint manifests.
4. Audit natural Search/Relay/Safety lifecycles. If real deployment fails, repair the observed cause under a separate bounded plan; do not redefine efficiency to hide it.
5. Broaden to30/60 and static/dynamic only once smoke gates pass, then verify >=15% on aggregate paired data and publish only necessary technical records.

Review focus: no-feasible-action cases, masked context rows, order-dependent IDs, physically impossible horizon assignments, reporting dwell as detection. No production baseline mode yet; comparator binding is independent benchmark tooling.

2026-10-10 next bounded diagnostic: benchmark-only cost-based candidate budgetK=4, SAMEfilter bothGPPO andrule. Keep allencoderrows/features intact; only reject outside4lowestcurrentlylegalobservedcost candidates perTask. Engineering choice based on prior4candidateprobe, not proven training-domain guarantee (checkpointdoesnotdeclareUAVtrainingcount). Candidate cutoff is a decisionbudget, NOT a physical infeasibility assertion. Development seed12345; holdoutseeds23456,34567,45678fixedbeforeexperiment. No productionfilter deployment or15claim unless completeH300holdout30/60staticdynamic evidence passes. Explicit physicalFREE T1coverage opt-in; frozen Searchactivationlegacy.

Extended Search deadline bound: mask ONLY geometrically proven impossible true-detection pairs. Lower bound=max(0,observed heat distance - public heat support radius - sensor range -sqrt(2)*physical cell size -velocity*maximum stale-position age)/velocity. No dwell time added, no obstacle/path-cost impossibility inference. Remaining horizon=frozen300s minuscurrenttime. Native path and Relay masks untouched; missing/nonfinite evidence leavespairunmasked. Regression boundary/stale/missing evidence beforeallocatorlink. Thiscannotprove15% policygain.

New heldout45678 failure: observed/physical8-connected component1535cells; heat1 minimumdistance41.503m andheat2 28.504m, generousdetectionallowance16.914m. Implementextended observation-only impossibilitygate: UNKNOWNpassable,8-neighborcomponents optimistic, publicframebounds; unionALLcomponents touchedbystale-position uncertainty; maskONLYifALLreachablecellregions lieoutsidepublicheat-support+sensor+cellallowance. No actualpathplanner, no truthmapinput. ReleaseactiveSearchpairONLYwithsameproof; retainunservicedheat andnormalallocatorrefill. Nativeuntouched.

Deploymenttrace confirms UNKNOWNcuts remain open; doNOTreplaceuncertaintywithoracletruth. Alternative design: observed Search route-support precedence. When at leastone CURRENTLYLEGALcandidate has a real-posecertifiedknownroute endingwithinpossiblepublicheat/sensor support (range+publicheatradius+cellallowance), defer candidates lackingthatobserved support; preservefullGPPOencoderfeatures andallcandidates ifnone supported. This is an admission preference, NOTproofphysicalinfeasibility/visibility/completion. Existing unsupported Search may yield ONLYif an available supportedalternativeexists; unservicedheatrefillsnormally. SAMEgates inrulebaseline. Reuseextendedtravel/provenprefixcertificate, no newpathplanner, no K4filterdeployment.
