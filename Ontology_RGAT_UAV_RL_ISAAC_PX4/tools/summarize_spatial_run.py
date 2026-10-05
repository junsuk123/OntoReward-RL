#!/usr/bin/env python3
"""Read completed spatial traces and write an explicitly descriptive report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ontology_rgat.spatial.reporting import trace_metrics,wilson_interval
from ontology_rgat.spatial.validation import actual_acceptance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--backend',choices=['local','isaac'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.run/'plan.json').read_text())
    matrix=json.loads((args.run/f'{args.backend}_evaluation.json').read_text())
    episodes=[]
    for row in matrix:
        path=args.run/f"{args.backend}_{row['mode']}_seed{row['seed']}.jsonl"
        groups={}
        for line in path.open():
            sample=json.loads(line)
            groups.setdefault(sample['seed'],[]).append(sample)
        for source in row['metrics']['rows']:
            metric=trace_metrics(groups.pop(source['seed']))
            if not metric['complete'] or metric['status']!=source['status']:
                raise ValueError('evaluation result and terminal trace differ')
            episodes.append(dict(mode=row['mode'],training_seed=row['seed'],
                episode_seed=source['seed'],episode_return=source['return'],
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metric))
        if groups:
            raise ValueError('trace has unreported or incomplete episodes')
    arms={}
    for mode in sorted({r['mode'] for r in matrix}):
        subset=[e for e in episodes if e['mode']==mode]
        successes=sum(e['status']=='SUCCESS' for e in subset)
        returns=[e['episode_return'] for e in subset]
        arms[mode]=dict(episodes=len(subset),successes=successes,
            landing_rate=successes/len(subset),
            landing_wilson_95_interval=wilson_interval(successes,len(subset)),
            mean_return=float(np.mean(returns)),median_return=float(np.median(returns)),
            independent_training_seeds=len({e['training_seed'] for e in subset}))
    report=dict(backend=args.backend,signature=plan['signature'],arms=arms,episodes=episodes,
        performance_parity_claim=False,checkpoint_selection=False,
        caveats=['Wilson intervals are episode-level descriptive intervals, not multi-training-seed uncertainty.',
                 'Optical availability/accepted track loss are not geometric FOV or false-detection labels.',
                 'Repeated development tests are not a fresh confirmatory test set.'])
    if args.backend=='isaac':
        report['acceptance']=actual_acceptance(matrix,plan['training_seeds'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(arms,indent=2))


if __name__=='__main__':
    main()
