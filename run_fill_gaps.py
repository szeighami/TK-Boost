"""Run missing experiments to get complete vanilla + NL-UDF coverage."""

import subprocess
import sys
import os

os.environ['PYTHONUNBUFFERED'] = '1'

# Instances missing vanilla runs
MISSING_VANILLA = [
    'local003', 'local004', 'local008', 'local029', 'local065',
    'local075', 'local131', 'local301', 'local1002', 'local1156',
]

# Instances missing NL-UDF runs
MISSING_NL_UDF = [
    'local017', 'local023', 'local039', 'local058', 'local066',
    'local163', 'local197', 'local210', 'local219', 'local309',
]

# Rerun gpt-5 runs with gpt-5.4 (existing runs used old model)
RERUN_NL_UDF_54 = [
    'local003', 'local004', 'local008', 'local029', 'local065',
    'local075', 'local131',
]
RERUN_VANILLA_54 = [
    'local017', 'local039', 'local058', 'local163', 'local197', 'local210',
]


def run(inst, mode, flags):
    print(f'\n{"="*70}')
    print(f'Running {inst} ({mode})...')
    print(f'{"="*70}')
    sys.stdout.flush()

    result = subprocess.run(
        [sys.executable, 'run_benchmark.py', '--instance', inst, '--max-turns', '25'] + flags,
        capture_output=False
    )

    status = 'done' if result.returncode == 0 else f'CRASHED (exit code {result.returncode})'
    print(f'{inst} ({mode}) {status}')
    sys.stdout.flush()


def main():
    total = len(MISSING_VANILLA) + len(MISSING_NL_UDF) + len(RERUN_NL_UDF_54) + len(RERUN_VANILLA_54)
    i = 0

    for inst in MISSING_VANILLA:
        i += 1
        print(f'\n[{i}/{total}]')
        run(inst, 'vanilla', ['--vanilla'])

    for inst in MISSING_NL_UDF:
        i += 1
        print(f'\n[{i}/{total}]')
        run(inst, 'NL-UDF', [])

    for inst in RERUN_NL_UDF_54:
        i += 1
        print(f'\n[{i}/{total}] (gpt-5.4 rerun)')
        run(inst, 'NL-UDF', [])

    for inst in RERUN_VANILLA_54:
        i += 1
        print(f'\n[{i}/{total}] (gpt-5.4 rerun)')
        run(inst, 'vanilla', ['--vanilla'])

    print(f'\n{"="*70}')
    print(f'COMPLETE: {total} runs')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
