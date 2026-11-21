"""Protect untouched final Search seeds from development use."""
FINAL_SEARCH_SEEDS=frozenset((101101,202202,303303))

def evaluation_stage(seed,uavs,horizon,final,domain):
    if domain not in ('static','dynamic'):raise ValueError('Unknown domain')
    if final:
        if seed not in FINAL_SEARCH_SEEDS or uavs not in (30,60) or horizon<300:
            raise ValueError('Final Search acceptance requires reserved seed, 30/60 UAVs and horizon >=300')
        return 'heldout final Search acceptance'
    if seed in FINAL_SEARCH_SEEDS:
        raise ValueError('Reserved final seeds cannot be consumed by development')
    return ('natural dynamic-obstacle development rollout, not final 15% acceptance' if domain=='dynamic'
            else 'development rollout, not final 15% acceptance')
