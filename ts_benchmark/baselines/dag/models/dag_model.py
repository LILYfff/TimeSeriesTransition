import torch
from torch import nn

from ts_benchmark.baselines.dag.layers.CC_EncDec import CovCausalityEncoder
from ts_benchmark.baselines.dag.layers.TC_EncDec import TemporalCausalityEncoder
from ts_benchmark.baselines.dag.layers.HCT import HistoricalTransitionFusion


class DAGModel(nn.Module):
    def __init__(self, config):
        super(DAGModel, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.use_c = config.use_c
        self.use_t = config.use_t
        self.use_c_exog = config.use_c_exog
        self.use_t_exog = config.use_t_exog
        self.alpha = config.alpha
        self.beta = config.beta
        self.series_dim = config.series_dim
        self.infer_use_future = config.infer_use_future

        self.hct_mode = int(getattr(config, "hct_mode", 0))
        self.hct_topk = int(getattr(config, "hct_topk", 5))
        self.hct_dim = int(getattr(config, "hct_dim", 128))
        self.hct_bins = int(getattr(config, "hct_bins", 24))
        self.hct_seed = int(getattr(config, "hct_seed", 2026))

        assert self.use_c or self.use_t, (
            "At least one of use_c or use_t must be True"
        )

        # ------------------------------------------------------------
        # IMPORTANT FOR FAIR B0/B1/B2 COMPARISON
        # ------------------------------------------------------------
        # Initialize the ORIGINAL DAG branches first, in exactly the same
        # order as hct_mode=0. HCT is attached only afterward.
        # ------------------------------------------------------------
        self.temporal_encoder = TemporalCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion,
        )

        self.cov_encoder = CovCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion,
        )

        # ------------------------------------------------------------
        # Attach HCT only after both original DAG branches exist.
        # ------------------------------------------------------------
        # HCT parameter initialization is isolated from the global PyTorch RNG.
        # Therefore enabling HCT does not change the random initialization of
        # the original DAG backbone or the later DataLoader shuffle state.
        # ------------------------------------------------------------
        if self.hct_mode > 0:
            cpu_rng_state = torch.get_rng_state()
            try:
                torch.manual_seed(self.hct_seed)
                hct_fusion = HistoricalTransitionFusion(
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    factor=config.factor,
                    dropout=config.dropout,
                    hct_dim=self.hct_dim,
                    bins=self.hct_bins,
                )
            finally:
                torch.set_rng_state(cpu_rng_state)

            self.temporal_encoder.set_hct_fusion(
                hct_fusion
            )

    def forward(self, input, exog_future, hct_pack=None):
        # input: [batch_size, seq_len, n_vars]
        temporal_causality_loss = 0
        cov_causality_loss = 0

        if not self.training and not self.infer_use_future:
            if self.use_t:
                (
                    t_output,
                    t_exog_output,
                    temporal_causality_loss,
                ) = self.temporal_encoder(
                    input,
                    None,
                    self.use_t_exog,
                    hct_pack=hct_pack,
                )

            if self.use_c:
                c_output, cov_causality_loss = self.cov_encoder(
                    input,
                    t_exog_output,
                    self.use_c_exog,
                )

            output = (
                self.alpha * t_output
                + (1 - self.alpha) * c_output
            )
            return output, 0

        else:
            if self.use_t:
                (
                    t_output,
                    _,
                    temporal_causality_loss,
                ) = self.temporal_encoder(
                    input,
                    exog_future,
                    self.use_t_exog,
                    hct_pack=hct_pack,
                )

            if self.use_c:
                c_output, cov_causality_loss = self.cov_encoder(
                    input,
                    exog_future,
                    self.use_c_exog,
                )

            output = (
                self.alpha * t_output
                + (1 - self.alpha) * c_output
            )

            causality_loss = self.beta * (
                temporal_causality_loss
                + cov_causality_loss
            )

            return output, causality_loss
