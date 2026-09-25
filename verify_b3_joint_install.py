import inspect
import torch

from ts_benchmark.baselines.dag.hct_retrieval import HCTRetriever


print("Imported HCTRetriever from:")
print(inspect.getfile(HCTRetriever))
print("\nConstructor signature:")
print(inspect.signature(HCTRetriever.__init__))

L = 4
H = 3
series_dim = 1
exo_dim = 1
D = series_dim + exo_dim
N = 40

# Synthetic memory.  Candidate j=8 is constructed to exactly match the
# current observable partial transition at query_start=28.
full = torch.randn(N, D) * 0.05

# Candidate j=8: A^(endo,exo)
A = torch.tensor([
    [0.2, -0.3],
    [0.4, -0.1],
    [0.6,  0.2],
    [0.8,  0.5],
], dtype=torch.float32)
B_exo = torch.tensor([[0.7], [0.9], [1.1]], dtype=torch.float32)
full[8:8+L] = A
full[8+L:8+L+H, series_dim:] = B_exo

# Query C is an exact copy of candidate A.
query_start = 28
full[query_start:query_start+L] = A
D_exo = B_exo.clone()

r3 = HCTRetriever(
    seq_len=L,
    pred_len=H,
    series_dim=series_dim,
    topk=1,
    stride=1,
    mode=3,
)
r3.prepare_memory(full)
pack3 = r3.retrieve_batch(
    full_series=full,
    query_starts=[query_start],
    future_exog=D_exo.unsqueeze(0),
)

assert bool(pack3["hist_valid"][0].item())
start3 = int(pack3["hist_start"][0, 0].item())
assert start3 == 8, f"B3-Joint expected start=8, got {start3}"
assert int(r3._candidate_ends[(r3._candidate_starts == start3).nonzero()[0]].item()) <= query_start

# B2 must not depend on future_exog and remains the old retrieval definition.
r2 = HCTRetriever(
    seq_len=L,
    pred_len=H,
    series_dim=series_dim,
    topk=1,
    stride=1,
    mode=2,
)
r2.prepare_memory(full)
p2a = r2.retrieve_batch(full, [query_start])
p2b = r2.retrieve_batch(full, [query_start], future_exog=torch.ones(1, H, exo_dim))
assert torch.equal(p2a["hist_start"], p2b["hist_start"])
assert torch.allclose(p2a["hist_score"], p2b["hist_score"])

print("\nB3-Joint synthetic retrieval: PASS")
print("selected historical start:", start3)
print("joint score:", float(pack3["hist_score"][0, 0].item()))
print("causal rule: PASS (B_end <= C_start)")
print("B2 future-exog independence: PASS")
print("\nB3 definition:")
print("ONE cosine on concat([A_endo,A_exo,B_exo]) vs concat([C_endo,C_exo,D_exo])")
print("No score averaging. No Top-M reranking. No lambda. D_endo is never passed.")
