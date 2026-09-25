import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.dag.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)


class TransitionEncoder(nn.Module):
    """
    Lightweight encoder for retrieved historical transitions A -> B.

    Inputs
    ------
    hist_src: [B, K, L, D]
        Historical endogenous context A.

    hist_fut: [B, K, H, D]
        Historical endogenous continuation B.

    Output
    ------
    transition_tokens: [B, K, D, hct_dim]
        One transition token per retrieved candidate and endogenous channel.

    Important normalization rule
    ----------------------------
    B is normalized with A's mean/std, not with B's own statistics. This
    preserves relative level shifts, amplitude changes, continuation/reversal,
    and other A -> B transition information.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        hct_dim=128,
        bins=24,
        dropout=0.0,
        eps=1e-5,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.hct_dim = int(hct_dim)
        self.bins = int(bins)
        self.eps = float(eps)

        if self.bins <= 0:
            raise ValueError("hct_bins must be a positive integer.")

        # Adaptive pooling makes the transition representation independent of
        # the exact L/H values and keeps the HCT branch lightweight.
        in_dim = 3 * self.bins

        self.transition_mlp = nn.Sequential(
            nn.Linear(in_dim, self.hct_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hct_dim, self.hct_dim),
            nn.LayerNorm(self.hct_dim),
        )

    def _pool_bins(self, x):
        """
        x: [B, K, T, D]
        -> [B, K, D, bins]
        """
        B, K, T, D = x.shape

        # Treat every (batch, candidate, endogenous channel) as one 1-D signal.
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * K * D, 1, T)
        x = F.adaptive_avg_pool1d(x, self.bins)
        x = x.view(B, K, D, self.bins)
        return x

    def forward(self, hist_src, hist_fut):
        if hist_src is None or hist_fut is None:
            raise ValueError("hist_src and hist_fut must not be None.")

        if hist_src.ndim != 4 or hist_fut.ndim != 4:
            raise ValueError(
                "HCT expects hist_src/hist_fut with shapes [B, K, L/H, D]."
            )

        if hist_src.shape[2] != self.seq_len:
            raise ValueError(
                f"HCT hist_src length {hist_src.shape[2]} != seq_len {self.seq_len}."
            )

        if hist_fut.shape[2] != self.pred_len:
            raise ValueError(
                f"HCT hist_fut length {hist_fut.shape[2]} != pred_len {self.pred_len}."
            )

        if hist_src.shape[0] != hist_fut.shape[0]:
            raise ValueError("hist_src and hist_fut batch sizes do not match.")

        if hist_src.shape[1] != hist_fut.shape[1]:
            raise ValueError("hist_src and hist_fut Top-K dimensions do not match.")

        if hist_src.shape[3] != hist_fut.shape[3]:
            raise ValueError("hist_src and hist_fut endogenous dimensions do not match.")

        # A statistics: [B, K, 1, D]
        src_mean = hist_src.mean(dim=2, keepdim=True)
        src_var = hist_src.var(dim=2, keepdim=True, unbiased=False)
        src_std = torch.sqrt(src_var + self.eps)

        # Normalize A and B with the SAME A statistics.
        src_norm = (hist_src - src_mean) / src_std
        fut_norm = (hist_fut - src_mean) / src_std

        # Coarse but shape-preserving temporal summaries.
        src_bins = self._pool_bins(src_norm)  # [B, K, D, bins]
        fut_bins = self._pool_bins(fut_norm)  # [B, K, D, bins]

        # Explicitly encode the historical state, the continuation, and the
        # coarse transition difference in the same pooled coordinate system.
        transition_input = torch.cat(
            [
                src_bins,
                fut_bins,
                fut_bins - src_bins,
            ],
            dim=-1,
        )  # [B, K, D, 3*bins]

        return self.transition_mlp(transition_input)


class HistoricalTransitionFusion(nn.Module):
    """
    Safe HCT residual adapter for DAG temporal tokens.

    current_tokens: [B * D, P, d_model]
    hist_src:       [B, K, L, D]
    hist_fut:       [B, K, H, D]
    hist_valid:     [B]

    Design choices
    --------------
    1. Historical transitions are represented in a small hct_dim space.
    2. Cross-attention is also performed in hct_dim rather than d_model.
    3. No LayerNorm is inserted on the original DAG temporal representation.
    4. A learnable residual scale starts EXACTLY at zero:

           out = current + tanh(scale) * gate * HCT_delta

       Therefore the first forward pass is exactly the original DAG temporal
       path. The model can gradually learn to use HCT only if it is helpful.
    5. Retrieval scores select Top-K only; they are not used as fusion weights.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        n_heads,
        factor=1,
        dropout=0.0,
        hct_dim=128,
        bins=24,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.hct_dim = int(hct_dim)

        self.transition_encoder = TransitionEncoder(
            seq_len=seq_len,
            pred_len=pred_len,
            hct_dim=self.hct_dim,
            bins=bins,
            dropout=dropout,
        )

        # Compress current DAG tokens before HCT cross-attention.
        self.current_projection = nn.Linear(
            self.d_model,
            self.hct_dim,
        )

        # Use the largest valid number of heads not exceeding the original
        # n_heads and dividing hct_dim exactly.
        hct_heads = min(int(n_heads), self.hct_dim)
        while hct_heads > 1 and self.hct_dim % hct_heads != 0:
            hct_heads -= 1

        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=False,
            ),
            d_model=self.hct_dim,
            n_heads=hct_heads,
        )

        self.output_projection = nn.Linear(
            self.hct_dim,
            self.d_model,
        )

        gate_hidden = max(16, self.hct_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hct_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # Critical safety mechanism: zero-init residual strength.
        # tanh(0) == 0, so the first forward pass is exactly current_tokens.
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        current_tokens,
        hist_src,
        hist_fut,
        hist_valid,
        series_dim,
    ):
        if hist_src is None or hist_fut is None or hist_valid is None:
            return current_tokens

        batch_size = hist_src.shape[0]
        series_dim = int(series_dim)

        if current_tokens.shape[0] != batch_size * series_dim:
            raise ValueError(
                "HCT current token batch mismatch: "
                f"current_tokens.shape[0]={current_tokens.shape[0]}, "
                f"batch_size={batch_size}, series_dim={series_dim}."
            )

        if hist_src.shape[-1] != series_dim:
            raise ValueError(
                "HCT endogenous channel mismatch: "
                f"hist_src.shape[-1]={hist_src.shape[-1]} != "
                f"series_dim={series_dim}."
            )

        # [B, K, D, hct_dim]
        hist_tokens = self.transition_encoder(
            hist_src=hist_src,
            hist_fut=hist_fut,
        )

        # Match DAG's flattened ordering: [B * D, K, hct_dim].
        hist_tokens = (
            hist_tokens.permute(0, 2, 1, 3)
            .contiguous()
            .view(
                batch_size * series_dim,
                hist_tokens.shape[1],
                self.hct_dim,
            )
        )

        # [B * D, P, hct_dim]
        current_small = self.current_projection(current_tokens)

        cross_small, _ = self.cross_attention(
            current_small,
            hist_tokens,
            hist_tokens,
            attn_mask=None,
        )

        # Scalar gate per current temporal patch token.
        gate = self.gate(
            torch.cat(
                [current_small, cross_small],
                dim=-1,
            )
        )  # [B * D, P, 1]

        # Project the HCT message back to DAG d_model.
        hct_delta = self.output_projection(cross_small)

        # Safe residual coefficient. Exactly zero at initialization.
        scale = torch.tanh(self.residual_scale)

        valid = hist_valid.to(
            device=current_tokens.device,
            dtype=current_tokens.dtype,
        )
        valid = valid.repeat_interleave(series_dim).view(-1, 1, 1)

        # No extra LayerNorm here: when scale=0, output is EXACTLY the
        # original DAG representation.
        return current_tokens + valid * scale * gate * hct_delta
