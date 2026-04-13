"""Run both vanilla and NL-UDF agents on all remaining local instances not yet tested."""

import subprocess
import sys
import os

os.environ['PYTHONUNBUFFERED'] = '1'

INSTANCES = [
    'local007', 'local009', 'local010', 'local018', 'local019', 'local021',
    'local022', 'local023', 'local024', 'local025', 'local026', 'local030',
    'local032', 'local034', 'local035', 'local038', 'local040', 'local041',
    'local050', 'local054', 'local055', 'local056', 'local059', 'local060',
    'local061', 'local062', 'local063', 'local064', 'local068', 'local070',
    'local072', 'local073', 'local074', 'local077', 'local078', 'local081',
    'local085', 'local096', 'local099', 'local100', 'local114', 'local128',
    'local130', 'local133', 'local141', 'local152', 'local157', 'local167',
    'local169', 'local170', 'local171', 'local193', 'local194', 'local195',
    'local196', 'local198', 'local199', 'local201', 'local202', 'local209',
    'local212', 'local220', 'local221', 'local229', 'local230', 'local244',
    'local253', 'local258', 'local259', 'local262', 'local263', 'local264',
    'local269', 'local270', 'local272', 'local273', 'local274', 'local275',
    'local277', 'local279', 'local284', 'local285', 'local286', 'local297',
    'local298', 'local299', 'local300', 'local302', 'local311', 'local329',
    'local330', 'local331', 'local335', 'local336', 'local344', 'local354',
    'local356', 'local360',
]

def main():
    total = len(INSTANCES)
    passed = 0
    failed = 0
    errors = 0

    for i, inst in enumerate(INSTANCES, 1):
        for mode, flags in [('vanilla', ['--vanilla']), ('NL-UDF', [])]:
            print(f'\n{"="*70}')
            print(f'[{i}/{total}] Running {inst} ({mode})...')
            print(f'{"="*70}')
            sys.stdout.flush()

            result = subprocess.run(
                [sys.executable, 'run_benchmark.py', '--instance', inst, '--max-turns', '25'] + flags,
                capture_output=False
            )

            if result.returncode != 0:
                errors += 1
                print(f'{inst} ({mode}) CRASHED (exit code {result.returncode})')
            else:
                print(f'{inst} ({mode}) done')

            sys.stdout.flush()

    print(f'\n{"="*70}')
    print(f'COMPLETE: {total} instances')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
