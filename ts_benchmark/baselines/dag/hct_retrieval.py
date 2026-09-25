import torch
import torch.nn.functional as F


class HCTRetriever:
    """
    Historical Conditional Transition Retriever.

    Modes
    -----
    mode=1 (B1): Endo-only retrieval
        S = sim(A_endo, C_endo)

    mode=2 (B2): Endo + Past Exo retrieval
        S = 0.5 * sim(A_endo, C_endo)
          + 0.5 * sim(A_exo,  C_exo)

    mode=3 (B3-Joint): Partial Joint Transition Retrieval

        Historical observable transition:
            O_hist = [A_endo, A_exo, B_exo]

        Current observable transition:
            O_cur  = [C_endo, C_exo, D_exo]

        A/C are the source/current windows, B/D are the future windows.
        D_exo is legal known-future exogenous information; D_endo is never
        provided to this retriever.

        The three observable parts are NOT scored separately and are NOT
        averaged or re-weighted.  They are concatenated in temporal order
        into one partial-joint-transition vector and ONE cosine similarity is
        computed:

            r_hist = normalize(vec([A_endo, A_exo, B_exo]))
            r_cur  = normalize(vec([C_endo, C_exo, D_exo]))
            S_B3   = r_hist @ r_cur

        The input data reaching this retriever are the same scaled values used
        by DAG.  Therefore B3 does not independently re-normalize each block,
        which would implicitly introduce block-specific weighting.  Only one
        final L2 normalization is applied to the complete joint vector.

    Important
    ---------
    1) The historical transition injected into the neural HCT module is
       ALWAYS endogenous: A_endo -> B_endo.
    2) Exogenous variables condition retrieval only.
    3) D_exo is allowed in B3 because it is known future exogenous input.
    4) D_endo is NEVER used for retrieval.
    5) Strict causal rule for indexed retrieval:
           B_end <= C_start.
    6) `bins`, `preselect`, and `future_lambda` are accepted only for backward
       compatibility with the previous experimental package.  B3-Joint does
       not use Top-M re-ranking or lambda weighting.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        series_dim,
        topk=5,
        stride=12,
        mode=1,
        bins=24,
        preselect=None,
        future_lambda=None,
        eps=1e-6,
    ):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.series_dim = int(series_dim)
        self.topk = int(topk)
        self.stride = int(stride)
        self.mode = int(mode)
        self.bins = int(bins)  # retained for compatibility / HCT config logging
        self.eps = float(eps)

        # Deprecated B3-two-stage arguments are intentionally ignored.
        self.preselect = preselect
        self.future_lambda = future_lambda

        if self.mode not in (1, 2, 3):
            raise NotImplementedError(
                "HCTRetriever supports hct_mode=1, 2, or 3."
            )
        if self.topk <= 0:
            raise ValueError("hct_topk must be > 0.")
        if self.stride <= 0:
            raise ValueError("hct_memory_stride must be > 0.")

        self._memory = None
        self._input_dim = None
        self._exo_dim = None

        self._candidate_starts = None
        self._candidate_ends = None
        self._candidate_src = None
        self._candidate_fut = None

        # B1/B2 features and B3 diagnostics.
        self._candidate_endo_feat = None
        self._candidate_past_exo_feat = None
        self._candidate_future_exo_feat = None

        # B3 unified partial-joint-transition feature.
        self._candidate_joint_feat = None

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _normalize_windows(self, x):
        """Per-window, per-channel z-normalization along time (B1/B2 only)."""
        mean = x.mean(dim=-2, keepdim=True)
        std = x.std(dim=-2, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)

    def _window_features(self, x):
        """B1/B2 feature: window z-normalize, flatten, then L2-normalize."""
        if x.ndim != 3:
            raise ValueError("HCT feature input must have shape [N, T, D].")
        if x.shape[-1] <= 0:
            raise ValueError("HCT feature input must contain at least one channel.")
        x = self._normalize_windows(x)
        x = x.reshape(x.shape[0], -1)
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

    def _raw_sequence_features(self, x):
        """
        Diagnostic-only feature for a raw/scaled sequence.

        No per-window re-normalization is performed.  The sequence is simply
        flattened and L2-normalized, consistent with the B3 joint geometry.
        """
        if x.ndim != 3:
            raise ValueError("Sequence input must have shape [N, T, D].")
        x = x.reshape(x.shape[0], -1)
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

    def _partial_joint_features(self, source_full, future_exog):
        """
        Build the B3 partial-joint-transition representation.

        Historical call:
            source_full = concat(A_endo, A_exo) = A^(endo,exo)
            future_exog = B_exo

        Current-query call:
            source_full = concat(C_endo, C_exo) = C^(endo,exo)
            future_exog = D_exo

        Both blocks are already on the same preprocessing scale as the DAG
        inputs.  We concatenate them BEFORE the only L2 normalization:

            feature = normalize(
                concat(flatten(source_full), flatten(future_exog))
            )

        This is ONE representation and ONE cosine similarity; there is no
        Endo/Past-Exo/Future-Exo score averaging and no lambda.
        """
        if source_full.ndim != 3 or future_exog.ndim != 3:
            raise ValueError(
                "B3 joint inputs must have shapes [N,L,D_all] and [N,H,D_exo]."
            )
        if source_full.shape[0] != future_exog.shape[0]:
            raise ValueError("B3 joint source/future batch sizes do not match.")
        if source_full.shape[1] != self.seq_len:
            raise ValueError(
                f"B3 source length {source_full.shape[1]} != seq_len {self.seq_len}."
            )
        if future_exog.shape[1] != self.pred_len:
            raise ValueError(
                f"B3 future-exo length {future_exog.shape[1]} != pred_len {self.pred_len}."
            )
        if source_full.shape[-1] != self._input_dim:
            raise ValueError(
                f"B3 source channel mismatch: {source_full.shape[-1]} != {self._input_dim}."
            )
        if future_exog.shape[-1] != self._exo_dim:
            raise ValueError(
                f"B3 future-exo channel mismatch: {future_exog.shape[-1]} != {self._exo_dim}."
            )

        source_flat = source_full.reshape(source_full.shape[0], -1)
        future_flat = future_exog.reshape(future_exog.shape[0], -1)
        joint = torch.cat([source_flat, future_flat], dim=-1)
        return F.normalize(joint, p=2, dim=-1, eps=self.eps)

    def _validate_exog(self, input_dim):
        exo_dim = int(input_dim) - self.series_dim
        if self.mode in (2, 3) and exo_dim <= 0:
            raise ValueError(
                f"hct_mode={self.mode} requires exogenous variables."
            )
        return exo_dim

    # ------------------------------------------------------------------
    # Candidate memory
    # ------------------------------------------------------------------
    def prepare_memory(self, full_series):
        """
        Prepare a frozen vectorized historical candidate bank.

        Candidate at start j:
            A = [j, j+L)
            B = [j+L, j+L+H)

        Only A_endo and B_endo are later injected into the HCT neural module.
        B3 retrieval uses the observable historical projection
            [A_endo, A_exo, B_exo].
        """
        if not torch.is_tensor(full_series):
            full_series = torch.as_tensor(full_series, dtype=torch.float32)

        full_series = full_series.detach().cpu().float().contiguous()
        if full_series.ndim != 2:
            raise ValueError("full_series must have shape [N, D_all].")
        if full_series.shape[-1] < self.series_dim:
            raise ValueError("full_series has fewer channels than series_dim.")

        self._memory = full_series
        self._input_dim = int(full_series.shape[-1])
        self._exo_dim = self._validate_exog(self._input_dim)

        N = int(full_series.shape[0])
        L = self.seq_len
        H = self.pred_len
        max_start = N - L - H

        if max_start < 0:
            self._candidate_starts = torch.empty(0, dtype=torch.long)
            self._candidate_ends = torch.empty(0, dtype=torch.long)
            self._candidate_src = torch.empty(
                0, L, self.series_dim, dtype=full_series.dtype
            )
            self._candidate_fut = torch.empty(
                0, H, self.series_dim, dtype=full_series.dtype
            )
            self._candidate_endo_feat = torch.empty(
                0, L * self.series_dim, dtype=full_series.dtype
            )
            self._candidate_past_exo_feat = None
            self._candidate_future_exo_feat = None
            self._candidate_joint_feat = None
            return

        starts = torch.arange(0, max_start + 1, self.stride, dtype=torch.long)

        source_full = torch.stack(
            [full_series[j:j + L, :] for j in starts.tolist()], dim=0
        )
        src_endo = source_full[..., :self.series_dim]
        fut_endo = torch.stack(
            [
                full_series[j + L:j + L + H, :self.series_dim]
                for j in starts.tolist()
            ],
            dim=0,
        )

        self._candidate_starts = starts
        self._candidate_ends = starts + L + H
        self._candidate_src = src_endo
        self._candidate_fut = fut_endo
        self._candidate_endo_feat = self._window_features(src_endo)

        if self.mode in (2, 3):
            src_exo = source_full[..., self.series_dim:]
            self._candidate_past_exo_feat = self._window_features(src_exo)
        else:
            src_exo = None
            self._candidate_past_exo_feat = None

        if self.mode == 3:
            fut_exo = torch.stack(
                [
                    full_series[j + L:j + L + H, self.series_dim:]
                    for j in starts.tolist()
                ],
                dim=0,
            )

            # Diagnostic-only direct B_exo feature.
            self._candidate_future_exo_feat = self._raw_sequence_features(fut_exo)

            # The actual B3 retrieval representation.
            self._candidate_joint_feat = self._partial_joint_features(
                source_full=source_full,
                future_exog=fut_exo,
            )
        else:
            self._candidate_future_exo_feat = None
            self._candidate_joint_feat = None

    def _ensure_memory(self, full_series):
        if self._memory is None:
            self.prepare_memory(full_series)
            return
        if tuple(full_series.shape) != tuple(self._memory.shape):
            self.prepare_memory(full_series)

    # ------------------------------------------------------------------
    # Query score construction
    # ------------------------------------------------------------------
    def _prepare_future_exog(self, future_exog, batch_size):
        if self.mode != 3:
            return None
        if future_exog is None:
            raise ValueError(
                "B3 / hct_mode=3 requires known future exogenous input "
                "future_exog=[B, pred_len, exo_dim]."
            )

        if not torch.is_tensor(future_exog):
            future_exog = torch.as_tensor(
                future_exog, dtype=self._memory.dtype
            )
        future_exog = future_exog.detach().cpu().float()
        if future_exog.ndim == 2:
            future_exog = future_exog.unsqueeze(0)

        expected = (int(batch_size), self.pred_len, self._exo_dim)
        if tuple(future_exog.shape) != expected:
            raise ValueError(
                "B3 future-exogenous shape mismatch. Expected "
                f"{expected}, received {tuple(future_exog.shape)}. "
                "Pass ONLY D_exo, never D_endo."
            )
        return future_exog

    def _score_contexts(self, contexts, future_exog=None):
        """Return final retrieval scores and diagnostic components [B,M]."""
        if contexts.ndim != 3:
            raise ValueError("query contexts must have shape [B, L, D].")
        if contexts.shape[1] != self.seq_len:
            raise ValueError(
                f"HCT context length mismatch: {contexts.shape[1]} != {self.seq_len}."
            )
        if contexts.shape[-1] < self.series_dim:
            raise ValueError("HCT query has fewer endogenous channels than series_dim.")

        query_endo = contexts[..., :self.series_dim]
        query_endo_feat = self._window_features(query_endo)
        score_endo = torch.matmul(
            query_endo_feat,
            self._candidate_endo_feat.transpose(0, 1),
        )
        components = {"endo": score_endo}

        if self.mode == 1:
            return score_endo, components

        query_exo = contexts[..., self.series_dim:]
        if query_exo.shape[-1] != self._exo_dim:
            raise ValueError(
                f"B{self.mode} query has {query_exo.shape[-1]} exogenous "
                f"channels, memory has {self._exo_dim}. Pass the FULL current "
                "history (Endo + Past Exo)."
            )

        query_past_exo_feat = self._window_features(query_exo)
        score_past_exo = torch.matmul(
            query_past_exo_feat,
            self._candidate_past_exo_feat.transpose(0, 1),
        )
        components["past_exo"] = score_past_exo

        # Keep B2 EXACTLY as the already-validated experiment.
        score_context = 0.5 * (score_endo + score_past_exo)
        components["context"] = score_context
        if self.mode == 2:
            return score_context, components

        # ------------------------------
        # B3-Joint: one unified similarity
        # ------------------------------
        future_exog = self._prepare_future_exog(
            future_exog=future_exog,
            batch_size=contexts.shape[0],
        )

        query_joint_feat = self._partial_joint_features(
            source_full=contexts,
            future_exog=future_exog,
        )
        score_joint = torch.matmul(
            query_joint_feat,
            self._candidate_joint_feat.transpose(0, 1),
        )

        # Direct future-exogenous similarity is logged only as a diagnostic;
        # it is NOT added to / averaged with the final B3 score.
        query_future_exo_feat = self._raw_sequence_features(future_exog)
        score_future_exo = torch.matmul(
            query_future_exo_feat,
            self._candidate_future_exo_feat.transpose(0, 1),
        )

        components["future_exo"] = score_future_exo
        components["joint"] = score_joint
        return score_joint, components

    # ------------------------------------------------------------------
    # Output pack helpers
    # ------------------------------------------------------------------
    def _empty_pack(self, batch_size):
        B = int(batch_size)
        K = self.topk
        dtype = self._memory.dtype

        pack = {
            "hist_src": torch.zeros(
                B, K, self.seq_len, self.series_dim, dtype=dtype
            ),
            "hist_fut": torch.zeros(
                B, K, self.pred_len, self.series_dim, dtype=dtype
            ),
            "hist_score": torch.zeros(B, K, dtype=dtype),
            "hist_score_endo": torch.zeros(B, K, dtype=dtype),
            "hist_valid": torch.zeros(B, dtype=torch.bool),
            "hist_start": torch.full((B, K), -1, dtype=torch.long),
        }

        if self.mode in (2, 3):
            pack["hist_score_past_exo"] = torch.zeros(B, K, dtype=dtype)
            pack["hist_score_context"] = torch.zeros(B, K, dtype=dtype)

        if self.mode == 3:
            pack["hist_score_future_exo"] = torch.zeros(B, K, dtype=dtype)
            pack["hist_score_joint"] = torch.zeros(B, K, dtype=dtype)

        return pack

    def _fill_selected(
        self,
        pack,
        b,
        selected_idx,
        selected_final_scores,
        component_scores,
    ):
        K = self.topk
        keep = int(selected_idx.numel())
        if keep == 0:
            return

        # Pad by repeating the last valid candidate if fewer than K exist.
        if keep < K:
            pad_count = K - keep
            selected_idx = torch.cat(
                [selected_idx, selected_idx[-1:].repeat(pad_count)], dim=0
            )
            selected_final_scores = torch.cat(
                [
                    selected_final_scores,
                    selected_final_scores[-1:].repeat(pad_count),
                ],
                dim=0,
            )

        pack["hist_src"][b] = self._candidate_src[selected_idx]
        pack["hist_fut"][b] = self._candidate_fut[selected_idx]
        pack["hist_score"][b] = selected_final_scores
        pack["hist_start"][b] = self._candidate_starts[selected_idx]
        pack["hist_valid"][b] = True
        pack["hist_score_endo"][b] = component_scores["endo"][b, selected_idx]

        if self.mode in (2, 3):
            pack["hist_score_past_exo"][b] = (
                component_scores["past_exo"][b, selected_idx]
            )
            pack["hist_score_context"][b] = (
                component_scores["context"][b, selected_idx]
            )

        if self.mode == 3:
            pack["hist_score_future_exo"][b] = (
                component_scores["future_exo"][b, selected_idx]
            )
            pack["hist_score_joint"][b] = (
                component_scores["joint"][b, selected_idx]
            )

    def _select_topk(self, scores, component_scores, causal_mask):
        B = scores.shape[0]
        K = self.topk
        pack = self._empty_pack(B)

        for b in range(B):
            valid_idx = torch.nonzero(causal_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue

            valid_scores = scores[b, valid_idx]
            keep = min(K, int(valid_idx.numel()))
            top_scores, local_idx = torch.topk(
                valid_scores, k=keep, largest=True, sorted=True
            )
            selected_idx = valid_idx[local_idx]

            self._fill_selected(
                pack=pack,
                b=b,
                selected_idx=selected_idx,
                selected_final_scores=top_scores,
                component_scores=component_scores,
            )

        return pack

    # ------------------------------------------------------------------
    # Public retrieval APIs
    # ------------------------------------------------------------------
    def retrieve_batch(
        self,
        full_series,
        query_starts,
        future_exog=None,
        max_b_end=None,
    ):
        """Vectorized index-based retrieval for training/validation."""
        self._ensure_memory(full_series)

        if torch.is_tensor(query_starts):
            query_starts = query_starts.detach().cpu().long()
        else:
            query_starts = torch.as_tensor(query_starts, dtype=torch.long)

        query_starts = query_starts.reshape(-1)
        B = query_starts.shape[0]
        N = self._memory.shape[0]
        L = self.seq_len

        contexts = torch.zeros(B, L, self._input_dim, dtype=self._memory.dtype)
        query_ok = (query_starts >= 0) & (query_starts + L <= N)

        for b, q in enumerate(query_starts.tolist()):
            if bool(query_ok[b].item()):
                contexts[b] = self._memory[q:q + L, :]

        scores, component_scores = self._score_contexts(
            contexts, future_exog=future_exog
        )

        if self._candidate_ends.numel() == 0:
            causal_mask = torch.zeros(B, 0, dtype=torch.bool)
        else:
            causal_mask = (
                self._candidate_ends.unsqueeze(0) <= query_starts.unsqueeze(1)
            )
            if max_b_end is not None:
                causal_mask = causal_mask & (
                    self._candidate_ends.unsqueeze(0) <= int(max_b_end)
                )
            causal_mask = causal_mask & query_ok.unsqueeze(1)

        return self._select_topk(
            scores=scores,
            component_scores=component_scores,
            causal_mask=causal_mask,
        )

    def retrieve_context_batch(
        self,
        full_series,
        query_contexts,
        max_b_end,
        future_exog=None,
    ):
        """Retrieval for forecasting when current C is supplied directly."""
        self._ensure_memory(full_series)

        if not torch.is_tensor(query_contexts):
            query_contexts = torch.as_tensor(
                query_contexts, dtype=self._memory.dtype
            )
        query_contexts = query_contexts.detach().cpu().float()
        if query_contexts.ndim == 2:
            query_contexts = query_contexts.unsqueeze(0)

        if query_contexts.shape[1] != self.seq_len:
            raise ValueError(
                f"HCT forecast context length mismatch: "
                f"{query_contexts.shape[1]} != {self.seq_len}."
            )

        if self.mode in (2, 3):
            if query_contexts.shape[-1] != self._input_dim:
                raise ValueError(
                    f"B{self.mode} forecast retrieval requires FULL current "
                    f"history. Expected {self._input_dim} channels, got "
                    f"{query_contexts.shape[-1]}."
                )
        elif query_contexts.shape[-1] < self.series_dim:
            raise ValueError("Forecast context has too few endogenous channels.")

        scores, component_scores = self._score_contexts(
            query_contexts,
            future_exog=future_exog,
        )

        B = query_contexts.shape[0]
        if self._candidate_ends.numel() == 0:
            causal_mask = torch.zeros(B, 0, dtype=torch.bool)
        else:
            allowed = self._candidate_ends <= int(max_b_end)
            causal_mask = allowed.unsqueeze(0).expand(B, -1)

        return self._select_topk(
            scores=scores,
            component_scores=component_scores,
            causal_mask=causal_mask,
        )

    # ------------------------------------------------------------------
    # Compatibility/debug APIs
    # ------------------------------------------------------------------
    def retrieve_tensors(self, full_series, query_start, future_exog=None):
        if self.mode == 3 and future_exog is not None:
            if torch.is_tensor(future_exog) and future_exog.ndim == 2:
                future_exog = future_exog.unsqueeze(0)

        pack = self.retrieve_batch(
            full_series=full_series,
            query_starts=[query_start],
            future_exog=future_exog,
        )
        return (
            pack["hist_src"][0],
            pack["hist_fut"][0],
            pack["hist_score"][0],
            bool(pack["hist_valid"][0].item()),
        )

    def retrieve(self, full_series, query_start, future_exog=None):
        if self.mode == 3 and future_exog is not None:
            if torch.is_tensor(future_exog) and future_exog.ndim == 2:
                future_exog = future_exog.unsqueeze(0)

        pack = self.retrieve_batch(
            full_series=full_series,
            query_starts=[query_start],
            future_exog=future_exog,
        )
        if not bool(pack["hist_valid"][0].item()):
            return None

        items = []
        for rank in range(self.topk):
            start = int(pack["hist_start"][0, rank].item())
            if start < 0:
                continue

            item = {
                "start": start,
                "a_start": start,
                "a_end": start + self.seq_len,
                "b_start": start + self.seq_len,
                "b_end": start + self.seq_len + self.pred_len,
                "score": float(pack["hist_score"][0, rank].item()),
                "score_endo": float(pack["hist_score_endo"][0, rank].item()),
                "A_endo": pack["hist_src"][0, rank],
                "B_endo": pack["hist_fut"][0, rank],
            }

            if self.mode in (2, 3):
                item["score_past_exo"] = float(
                    pack["hist_score_past_exo"][0, rank].item()
                )
                item["score_context"] = float(
                    pack["hist_score_context"][0, rank].item()
                )

            if self.mode == 3:
                item["score_future_exo_diagnostic"] = float(
                    pack["hist_score_future_exo"][0, rank].item()
                )
                item["score_joint"] = float(
                    pack["hist_score_joint"][0, rank].item()
                )

            items.append(item)
        return items
