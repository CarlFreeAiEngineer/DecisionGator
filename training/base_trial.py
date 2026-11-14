"""Try another foundation model with the unchanged 0.4 recipe and score it uncompressed.

    uv run training/base_trial.py --name NAME --base HUGGINGFACE_MODEL --revision COMMIT

Trains like training/retrain.py without corrections, exports the float model, fits the
temperature, and scores every held-out test. Compare with the 0.4.0 column of
reports/accuracy-v4.1.md (0.4.0 is the uncompressed model). Compression is left out on
purpose: training/quant/mixed_quant.py is tuned for DeBERTa-v3-large.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import pipeline  # noqa: E402
from training.retrain import TESTS, TRAINING_DATA, percent, score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='run folder under runs/')
    parser.add_argument('--base', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='cuda')
    args = parser.parse_args()

    run = ROOT / 'runs' / args.name
    extra = [str(f) for f in TRAINING_DATA]
    started = time.monotonic()
    pipeline.train(SimpleNamespace(output=str(run / 'checkpoint'), extra_data=extra, data=None, epochs=args.epochs,
                                   batch_size=args.batch_size, learning_rate=args.learning_rate, seed=42, device=args.device,
                                   start=None, resume=False, base=args.base, revision=args.revision, nli_head=True, template=2,
                                   gradient_checkpointing=False, fixed_shapes=False))
    bundle = run / 'float'
    pipeline.export(SimpleNamespace(checkpoint=str(run / 'checkpoint/best'), output=str(bundle), model_id=f'trial-{args.name}'))
    pipeline.evaluate(SimpleNamespace(command='calibrate', bundle=str(bundle), data=None, extra_data=extra,
                                      split='calibration', output=str(run / 'calibration.json')))
    results = {label: percent(score(bundle, files, 'test', run / f'test-{label.lower().replace(" ", "-")}.json'))
               for label, files in TESTS.items()}
    results['minutes'] = round((time.monotonic() - started) / 60)
    (run / 'summary.json').write_text(json.dumps({'base': args.base, 'revision': args.revision, **results}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
