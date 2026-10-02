"""Resume interrupted EM training before completing its manifest pipeline row."""
import argparse
import json
import shlex
from pathlib import Path

from dalg.pipeline import (
    _ensure_run_spec,
    _run_command,
    _training_command,
    execute_run,
    read_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('index', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    runs = read_manifest(args.manifest)
    if not 0 <= args.index < len(runs):
        raise ValueError('Manifest index out of bounds')
    run = runs[args.index]
    directory = Path(run['run_dir'])
    if run['training']['model_kind'] != 'hddc' or run['training']['arguments']['fit_method'] != 'em':
        raise ValueError('This retry worker requires HDDC EM')
    if not (directory / 'TRAINING_COMPLETED.json').exists():
        if not (directory / 'checkpoint.pt').is_file():
            raise FileNotFoundError(directory / 'checkpoint.pt')
        history = json.loads((directory / 'em_history.json').read_text())
        print(f"Resume row {args.index} after iteration {history['history'][-1]['iteration']}", flush=True)
        command = _training_command(run)
        if args.dry_run:
            print(shlex.join(command))
        else:
            _ensure_run_spec(run)
            _run_command(command)
    if args.dry_run:
        print(f"Then complete assignments and evaluation for {run['run_id']}")
    else:
        execute_run(run)


if __name__ == '__main__':
    main()
