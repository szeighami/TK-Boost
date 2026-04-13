"""Run NL UDF agent on all local instances with ground truth (excluding already-tested ones)."""

import json
import csv
import os
import sys
import time
from pathlib import Path

import tkboost
from tkboost import SQLAgent, SQLiteExecutor

# Instances to skip (already tested or excluded by user)
EXCLUDE = {
    'local007', 'local008', 'local015', 'local019', 'local020', 'local022',
    'local023', 'local026', 'local030', 'local031', 'local038', 'local054',
    'local056', 'local064', 'local067', 'local078', 'local097', 'local099',
    'local130', 'local198', 'local199', 'local201', 'local202', 'local229',
    'local274', 'local300', 'local329', 'local358',
    # Already tested
    'local003', 'local004', 'local017', 'local029', 'local039', 'local058',
    'local065', 'local066', 'local075', 'local131', 'local163', 'local197',
    'local210',
}

# DB name mapping for mismatched filenames
DB_NAME_MAP = {
    'SQLITE_SAKILA': 'sqlite-sakila',
    'DB_IMDB': 'Db-IMDB',
}

JSONL_PATH = 'data/spider2-lite.jsonl'
GT_SQL_DIR = 'evaluation/gold/sql'
GT_RESULT_DIR = 'evaluation/gold/exec_result'
DB_DIR = 'data/spider2'
RESULTS_CSV = 'benchmark_results.csv'


def resolve_db_path(db_name):
    """Resolve the SQLite database path for a given db name."""
    # Try direct match first
    path = os.path.join(DB_DIR, f'{db_name}.sqlite')
    if os.path.exists(path):
        return path
    # Try mapped name
    mapped = DB_NAME_MAP.get(db_name)
    if mapped:
        path = os.path.join(DB_DIR, f'{mapped}.sqlite')
        if os.path.exists(path):
            return path
    return None


def load_gt_result(instance_id):
    """Load ground truth result CSV (tries _a variant first, then plain)."""
    for suffix in ['_a', '']:
        path = os.path.join(GT_RESULT_DIR, f'{instance_id}{suffix}.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = [row for row in reader]
            return headers, rows
    return None, None


def load_gt_column_names(instance_id):
    """Load column names from all GT result CSV variants (_a, _b, etc. and plain)."""
    all_col_names = []
    # Check for _a, _b, ... variants
    for suffix_char in 'abcdefgh':
        path = os.path.join(GT_RESULT_DIR, f'{instance_id}_{suffix_char}.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                all_col_names.append(headers)
    # Also check plain (no suffix)
    if not all_col_names:
        path = os.path.join(GT_RESULT_DIR, f'{instance_id}.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                all_col_names.append(headers)
    return all_col_names


def build_expected_output_format(instance_id):
    """Build expected output format string from GT column names."""
    all_col_names = load_gt_column_names(instance_id)
    if not all_col_names:
        return None
    if len(all_col_names) == 1:
        return f"Expected Output Format: columns={all_col_names[0]} (use this exact order)."
    variants_str = "\n".join([f"  Option {i+1}: {cols}" for i, cols in enumerate(all_col_names)])
    return f"Expected Output Format (multiple valid options):\n{variants_str}\n(Choose one option and use that exact column order)."


def compare_results(agent_rows, gt_headers, gt_rows):
    """Simple comparison: check if agent result values match ground truth."""
    if not agent_rows or not gt_rows:
        return False
    # Convert both to sets of tuples for order-independent comparison
    agent_set = {tuple(str(v) for v in row) for row in agent_rows}
    gt_set = {tuple(row) for row in gt_rows}
    return agent_set == gt_set


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run NL UDF agent benchmark')
    parser.add_argument('--instance', type=str, default=None,
                        help='Run a single instance by ID (e.g. local003)')
    parser.add_argument('--vanilla', action='store_true',
                        help='Run vanilla agent (no NL UDF sub-agents)')
    parser.add_argument('--max-turns', type=int, default=15,
                        help='Max turns for main agent (default: 15)')
    args = parser.parse_args()

    cfg = tkboost.init(provider='auto')
    print(f'Provider: {cfg["provider"]}  |  Model: {cfg["model"]}')

    # Load all instances from JSONL
    instances = {}
    with open(JSONL_PATH) as f:
        for line in f:
            inst = json.loads(line)
            instances[inst['instance_id']] = inst

    if args.instance:
        # Run a single instance
        candidates = [args.instance]
    else:
        # Find eligible instances: have ground truth SQL and not excluded
        gt_ids = {f.replace('.sql', '') for f in os.listdir(GT_SQL_DIR) if f.startswith('local')}
        candidates = sorted(gt_ids - EXCLUDE)

    print(f'\nRunning {len(candidates)} instances: {candidates}\n')

    results = []
    for instance_id in candidates:
        inst = instances.get(instance_id)
        if not inst:
            print(f'[SKIP] {instance_id}: not found in JSONL')
            continue

        db_name = inst['db']
        db_path = resolve_db_path(db_name)
        if not db_path:
            print(f'[SKIP] {instance_id}: database {db_name} not found')
            results.append({
                'instance_id': instance_id, 'db': db_name,
                'status': 'skipped', 'reason': 'db_not_found',
                'agent_result': None, 'gt_result': None, 'match': None,
            })
            continue

        print(f'\n{"="*70}')
        print(f'Running {instance_id} [{db_name}]')
        print(f'Q: {inst["question"][:120]}...')
        print(f'{"="*70}')

        try:
            expected_output_format = build_expected_output_format(instance_id)
            if expected_output_format:
                print(f'🧾 {expected_output_format}')

            executor = SQLiteExecutor(db_path)
            agent = SQLAgent(max_turns=args.max_turns, verbose=True, vanilla=args.vanilla)
            result = agent.translate(
                question=inst['question'],
                executor=executor,
                db_name=db_name,
                instance_id=instance_id,
                external_knowledge=inst.get('external_knowledge'),
                expected_output_format=expected_output_format,
            )

            agent_rows = result.get('preview_rows', [])
            gt_headers, gt_rows = load_gt_result(instance_id)
            match = compare_results(agent_rows, gt_headers, gt_rows)

            print(f'\n--- {instance_id} ---')
            print(f'Agent:  {agent_rows[:3]}')
            if gt_rows:
                print(f'GT:     {gt_rows[:3]}')
            print(f'Match:  {match}')
            print(f'Trace:  {result.get("trace_path")}')

            results.append({
                'instance_id': instance_id, 'db': db_name,
                'status': 'ok', 'reason': '',
                'agent_result': str(agent_rows[:3]),
                'gt_result': str(gt_rows[:3]) if gt_rows else None,
                'match': match,
                'trace_path': result.get('trace_path'),
            })

        except Exception as e:
            print(f'\n[ERROR] {instance_id}: {e}')
            results.append({
                'instance_id': instance_id, 'db': db_name,
                'status': 'error', 'reason': str(e),
                'agent_result': None, 'gt_result': None, 'match': False,
            })

    # Write results CSV
    print(f'\n\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')

    total = len(results)
    ok = sum(1 for r in results if r['status'] == 'ok')
    matched = sum(1 for r in results if r.get('match') is True)
    failed = sum(1 for r in results if r['status'] == 'error')
    skipped = sum(1 for r in results if r['status'] == 'skipped')

    print(f'Total: {total}  |  Ran: {ok}  |  Matched GT: {matched}/{ok}  |  Errors: {failed}  |  Skipped: {skipped}')
    print()
    for r in results:
        status = 'MATCH' if r.get('match') else ('ERROR' if r['status'] == 'error' else ('SKIP' if r['status'] == 'skipped' else 'MISMATCH'))
        print(f"  {r['instance_id']:12s} [{r['db']:30s}]  {status}")

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instance_id', 'db', 'status', 'reason', 'agent_result', 'gt_result', 'match', 'trace_path'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {RESULTS_CSV}')


if __name__ == '__main__':
    main()
