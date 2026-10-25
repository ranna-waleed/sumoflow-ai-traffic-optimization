"""Deprecated compatibility entry point for the former fair evaluator.

The previous implementation mixed a historical JSON baseline with newly
generated XML and silently accepted untrained agents. Use
``python -m DeepQN.evaluation.evaluate`` with an explicit controller, seed,
checkpoint episode, and paired baseline manifests instead.
"""

from __future__ import annotations

from DeepQN.evaluation.evaluate import evaluate


def compare_fair(*args, **kwargs):
    raise RuntimeError(
        "evaluate_fair is retired because it can create scientifically invalid "
        "comparisons; use DeepQN.evaluation.evaluate.evaluate with paired manifests"
    )


if __name__ == "__main__":
    raise SystemExit(
        "This evaluator is retired. Run: python -m DeepQN.evaluation.evaluate --help"
    )
