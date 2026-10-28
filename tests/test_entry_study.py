"""Stage I (lab/entry_study.py, DECISIONS 87): registered menu, random block and verdict rules."""
import numpy as np
import pandas as pd

from lab import blocks, entry_study as es
from tests.test_stage_c import _study_panel


def test_registered():
    assert len(es.ENTRY_KEYS) == 15 and es.RANDOM_SEEDS == [1, 2, 3, 4, 5] and es.N_ORDERS == 100
    assert es.EXITS_I == {"H1": "H1", "H9": "X9"} and es.HALVES == ("1993-2007", "2008-2023") and es.N_NEW == 90


def test_random_block_rate_and_seed():
    p, _ = _study_panel()
    a = blocks.ALL_BLOCKS["g_random"].compute(p, 1)
    assert a.equals(blocks.ALL_BLOCKS["g_random"].compute(p, 1)) and not a.equals(blocks.ALL_BLOCKS["g_random"].compute(p, 2))
    assert 0.005 < a.to_numpy().mean() < 0.015


def _fake(buy, exit_, k, cagr_a, cagr_b, contrib, n=3):
    acc = [{"buy": buy, "exit": exit_, "k": k, "order": f"random_{o}", "trades": 50,
            **{f"cagr_{p}": (cagr_a if p == "1993-2007" else cagr_b) for p in es.PERIODS},
            **{f"mdd_{p}": -0.2 for p in es.PERIODS}} for o in range(n)]
    un = {"buy": buy, "exit": exit_, "k": k, "signals": 100, "capped_share": 0.8,
          **{f"mean_contrib_{p}": contrib for p in es.PERIODS}, **{f"n_{p}": 100 for p in es.PERIODS},
          **{f"big_{p}": 0.05 for p in es.PERIODS}, **{f"mean_x_spy_{p}": 0.0 for p in es.PERIODS}}
    return acc, un


def test_verdict_rules(monkeypatch):
    monkeypatch.setattr(es, "ENTRY_KEYS", ["E1"])
    acc, un = [], []
    for k in es.K_VALUES:
        for b, ca, cb, c in (("E1", 0.10, 0.08 if k != 3.0 else 0.01, 0.02), ("E0s1", 0.05, 0.05, 0.01)):
            a_, u_ = _fake(b, "H1", k, ca, cb, c)
            acc += a_
            un.append(u_)
    base = pd.DataFrame([{"baseline": "SPY", **{f"cagr_{p}": 0.07 for p in es.PERIODS}, **{f"mdd_{p}": -0.5 for p in es.PERIODS}}])
    v = es.verdicts(pd.DataFrame(acc), pd.DataFrame(un), base)
    assert list(v["b_halves"]) == [True, True, False] and v["meaningful_b"].all()  # 2 of 3 k
    assert v["meaningful_a"].all() and (v["pct_in_e0_1993-2007"] == 1.0).all()
    assert list(v["market_pass"]) == [True, True, False] and v["candidate_20bp"].tolist() == [True, True, False]


def test_random_block_not_in_app_menus():
    assert "g_random" not in blocks.ENTRY_BLOCKS and "g_random" in blocks.ALL_BLOCKS
