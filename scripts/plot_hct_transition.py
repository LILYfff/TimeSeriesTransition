import numpy as np
import matplotlib.pyplot as plt


data = np.load(
    "./hct_debug/b1_retrieval_check.npz"
)

C = data["C_endo"][:, 0]
D = data["D_endo"][:, 0]

L = len(C)
H = len(D)


def normalize_transition(src, fut):
    """
    用 src 的统计量同时归一化 src 和 fut。

    非常重要：
    不能分别标准化 src 和 fut，
    否则会破坏真正的 transition 信息。
    """
    mean = src.mean()
    std = src.std() + 1e-6

    src_n = (src - mean) / std
    fut_n = (fut - mean) / std

    return src_n, fut_n


# Current C -> D
C_n, D_n = normalize_transition(C, D)

current_transition = np.concatenate(
    [C_n, D_n]
)


plt.figure(figsize=(14, 7))

x = np.arange(L + H)

plt.plot(
    x,
    current_transition,
    linewidth=3,
    label="Current C -> D (ground truth)",
)


# Historical A -> B
for rank in range(1, 6):

    A = data[f"A{rank}_endo"][:, 0]
    B = data[f"B{rank}_endo"][:, 0]

    A_n, B_n = normalize_transition(A, B)

    hist_transition = np.concatenate(
        [A_n, B_n]
    )

    score = float(
        data[f"score{rank}"][0]
    )

    plt.plot(
        x,
        hist_transition,
        linewidth=1.3,
        label=f"Top-{rank} A->B, context sim={score:.3f}",
    )


# Boundary between history and future
plt.axvline(
    x=L - 1,
    linestyle="--",
    linewidth=2,
)

plt.text(
    L - 8,
    plt.ylim()[1] * 0.9,
    "A/C",
)

plt.text(
    L + 3,
    plt.ylim()[1] * 0.9,
    "B/D",
)

plt.xlabel("Time step")
plt.ylabel("Normalized endogenous value")

plt.title(
    "B1 Historical Transition Check: "
    "A->B vs Current C->D"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "./hct_debug/b1_transition_check.png",
    dpi=300,
)

plt.show()