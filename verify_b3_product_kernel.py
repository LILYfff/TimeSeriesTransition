#!/usr/bin/env python3
import torch
from ts_benchmark.baselines.dag.hct_retrieval import HCTRetriever


def main():
    L = 4
    H = 4
    series_dim = 1
    x = torch.zeros(60, 2)

    c_endo = torch.tensor([0.0, 1.0, 2.0, 3.0])
    c_exo = torch.tensor([0.0, 1.0, 0.0, -1.0])
    d_exo = torch.tensor([2.0, 3.0, 2.0, 1.0])
    x[24:28, 0] = c_endo
    x[24:28, 1] = c_exo

    # Candidate 0: perfect B2 context, incompatible future-exogenous evolution.
    x[0:4, 0] = c_endo
    x[0:4, 1] = c_exo
    x[4:8, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x[4:8, 1] = torch.tensor([-2.0, -3.0, -2.0, -1.0])

    # Candidate 1: almost identical context, matching future-exogenous evolution.
    x[12:16, 0] = torch.tensor([0.0, 1.1, 2.0, 3.0])
    x[12:16, 1] = c_exo
    x[16:20, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x[16:20, 1] = d_exo

    b2 = HCTRetriever(
        seq_len=L, pred_len=H, series_dim=series_dim,
        topk=1, stride=12, mode=2, bins=4,
    )
    b2.prepare_memory(x)
    p2 = b2.retrieve_batch(x, [24])
    assert int(p2["hist_start"][0, 0]) == 0

    b3 = HCTRetriever(
        seq_len=L, pred_len=H, series_dim=series_dim,
        topk=1, stride=12, mode=3, bins=4,
    )
    b3.prepare_memory(x)
    p3 = b3.retrieve_batch(
        x,
        [24],
        future_exog=d_exo.view(1, H, 1),
    )
    assert int(p3["hist_start"][0, 0]) == 12
    assert int(p3["hist_start"][0, 0]) + L + H <= 24
    assert "hist_score_drive" in p3
    assert "hist_kernel_context" in p3
    assert "hist_kernel_drive" in p3

    print("B3 product-kernel synthetic test: PASS")
    print("B2 selected start:", int(p2["hist_start"][0, 0]))
    print("B3 selected start:", int(p3["hist_start"][0, 0]))
    print("B3 context score:", float(p3["hist_score_context"][0, 0]))
    print("B3 drive score  :", float(p3["hist_score_drive"][0, 0]))
    print("B3 product score:", float(p3["hist_score"][0, 0]))
    print("Causal rule: PASS (B_end <= C_start)")
    print("D_endo is not an API input to B3 retrieval.")


if __name__ == "__main__":
    main()
