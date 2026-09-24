# -*- coding: utf-8 -*-
"""A/B: honest_eval (exp_sweep) vs eval_strategy (exp_thresh) на одной конфигурации v3.

Проверяет, одинаково ли считают macro-F1 два харнесса при одной стратегии порога.
"""
import numpy as np

import _common  # noqa: F401

from src import config as C
from src import dataset as ds
from src import features as fl
from exp_sweep import CriterionScores, honest_eval
from build_v3 import V3_FEATURES, V3_SOURCES
from exp_thresh import eval_strategy, THR

data = fl.load_features(ds.load_manifest(C.MANIFEST_CSV))
real = np.asarray(ds.real_mask(data))
cs = CriterionScores(data, real)
S = {n: cs.get(n, V3_FEATURES[n], V3_SOURCES[n]) for n in C.VIOLATIONS}

for ns in (12, 20):
    h = honest_eval(data, real, S, n_seeds=ns)
    e = eval_strategy(data, real, S, {n: THR["f1"] for n in C.VIOLATIONS}, n_seeds=ns)
    print("seeds=%2d  honest_eval=%.4f  eval_strategy(f1)=%.4f" % (ns, h["macro"], e["macro"]))
    print("   honest per-label:", {k: round(v, 3) for k, v in h["per_label"].items()})
    print("   eval   per-label:", {k: round(v, 3) for k, v in e["per_label"].items()})
