"""Run all local instances with gpt-5.4, both vanilla and NL-UDF.

Skips any instance+mode that already has a gpt-5.4 trace (timestamp >= 20260411_070000).
Dynamically checks what's already done at startup, so safe to restart after interruption.

Usage:
  cd TK-Boost && source /home/sep/projects/set_api_keys && python run_all_gpt54.py
"""

import subprocess
import sys
import os
import re
import json

os.environ['PYTHONUNBUFFERED'] = '1'

DB_NAME_MAP = {'SQLITE_SAKILA': 'sqlite-sakila', 'DB_IMDB': 'Db-IMDB'}
GPT54_CUTOFF = '20260411_070000'


def get_all_local_instances():
    instances = []
    with open('data/spider2-lite.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj['instance_id']
            if not iid.startswith('local'):
                continue
            db = obj['db']
            db_path = f'data/spider2/{db}.sqlite'
            mapped = DB_NAME_MAP.get(db)
            if mapped:
                db_path = f'data/spider2/{mapped}.sqlite'
            if os.path.exists(db_path):
                instances.append(iid)
    return sorted(set(instances))


def get_done_gpt54():
    """Return sets of (vanilla_done, nl_done) instance IDs with gpt-5.4 traces."""
    vanilla_done = set()
    nl_done = set()
    traces_dir = 'traces'
    if not os.path.isdir(traces_dir):
        return vanilla_done, nl_done
    for entry in os.listdir(traces_dir):
        entry_path = os.path.join(traces_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        m = re.match(r'^(local\w+)_(\d{8}_\d{6})$', entry)
        if not m:
            continue
        inst, ts = m.group(1), m.group(2)
        if ts < GPT54_CUTOFF:
            continue
        has_sub = os.path.isdir(os.path.join(entry_path, 'sub_agents'))
        if has_sub:
            nl_done.add(inst)
        else:
            vanilla_done.add(inst)
    return vanilla_done, nl_done


def run(inst, mode, flags):
    print(f'  Running {inst} ({mode})...', end=' ', flush=True)
    result = subprocess.run(
        [sys.executable, 'run_benchmark.py', '--instance', inst, '--max-turns', '25'] + flags,
        capture_output=True
    )
    if result.returncode != 0:
        print(f'CRASHED (exit code {result.returncode})')
    else:
        print('done')
    sys.stdout.flush()
    return result.returncode


def main():
    all_ids = get_all_local_instances()
    vanilla_done, nl_done = get_done_gpt54()

    need_vanilla = [i for i in all_ids if i not in vanilla_done]
    need_nl = [i for i in all_ids if i not in nl_done]

    total = len(need_vanilla) + len(need_nl)
    print(f'Total instances: {len(all_ids)}')
    print(f'Already done (gpt-5.4): {len(vanilla_done)} vanilla, {len(nl_done)} NL-UDF')
    print(f'Runs needed: {len(need_vanilla)} vanilla + {len(need_nl)} NL-UDF = {total}')
    print()

    i = 0
    for inst in need_vanilla:
        i += 1
        print(f'[{i}/{total}]')
        run(inst, 'vanilla', ['--vanilla'])

    for inst in need_nl:
        i += 1
        print(f'[{i}/{total}]')
        run(inst, 'NL-UDF', [])

    print(f'\n{"="*70}')
    print(f'COMPLETE: {total} runs')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
