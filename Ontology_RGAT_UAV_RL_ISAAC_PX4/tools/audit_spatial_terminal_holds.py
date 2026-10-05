#!/usr/bin/env python3
"""Read saved traces without changing outcomes or selecting checkpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from ontology_rgat.spatial.auditing import terminal_hold_audit


def audit(paths):
    rows = []
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                info = row.get('info', {})
                if info.get('status') == 'SAFE_ABORT':
                    rows.append(dict(trace=str(path), seed=row.get('seed'),
                                     mode=row.get('mode', row.get('controller')),
                                     recorded_outcome=info['status'],
                                     **terminal_hold_audit(info)))
    return dict(audit='terminal-hold-snapshot/1', outcomes_relabelled=False,
                checkpoint_selection=False, sustained_stability_claim=False,
                aborts=len(rows), evaluated=sum(r['evaluated'] for r in rows),
                verified=sum(r['hold_verified'] for r in rows),
                failed_checks=dict(Counter(k for r in rows
                    for k,v in r.get('checks', {}).items() if not v)), rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=audit(args.traces)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        # Historical evidence is immutable; a separate new audit is explicit.
        with args.output.open('x') as stream:
            stream.write(text)
    print(text)


if __name__=='__main__':
    main()
