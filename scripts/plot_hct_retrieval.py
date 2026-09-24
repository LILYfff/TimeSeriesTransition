import numpy as np
import matplotlib.pyplot as plt


data = np.load(
    "./hct_debug/b1_retrieval_check.npz"
)

C = data["C_endo"]

# 目前 ETTh1 target_channel=[-1]
# series_dim 应该是 1
C = C[:, 0]

plt.figure(figsize=(12, 6))

# 当前 C
C_norm = (
    C - C.mean()
) / (
    C.std() + 1e-6
)

plt.plot(
    np.arange(len(C)),
    C_norm,
    linewidth=3,
    label="Current C",
)

# Top-5 historical A
for rank in range(1, 6):

    A = data[f"A{rank}_endo"][:, 0]

    A_norm = (
        A - A.mean()
    ) / (
        A.std() + 1e-6
    )

    score = float(
        data[f"score{rank}"][0]
    )

    plt.plot(
        np.arange(len(A)),
        A_norm,
        linewidth=1.5,
        label=f"Top-{rank} A, sim={score:.3f}",
    )

plt.xlabel("Time step")
plt.ylabel("Normalized endogenous value")
plt.title(
    "B1 Retrieval Check: "
    "Current C vs Historical A"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "./hct_debug/b1_context_similarity.png",
    dpi=300
)

plt.show()