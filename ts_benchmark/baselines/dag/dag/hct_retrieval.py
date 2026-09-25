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
        S_ctx = 0.5 * sim(A_endo, C_endo)
              + 0.5 * sim(A_exo,  C_exo)

    mode=3 (B3): two-stage conditional retrieval
        Stage 1: use the exact B2 context score S_ctx to preselect Top-M
        causal historical transitions.

        Stage 2: within those Top-M candidates, compare the exogenous
        TRANSITION pattern
            A_exo -> B_exo
        against the currently known exogenous transition
            C_exo -> D_exo.

        The final re-ranking score is
            S_B3 = Z(S_ctx) + lambda * Z(S_future_transition)

        where Z(.) is a per-query z-score computed only inside the
        preselected Top-M candidate set.

    Important
    ---------
    1) The historical transition injected into the neural HCT module is
       ALWAYS endogenous: A_endo -> B_endo.
    2) Exogenous variables condition retrieval only.
    3) D_exo is allowed in B3 because it is known future exogenous input.
    4) D_endo is NEVER used for retrieval.
    5) Strict causal rule for indexed retrieval:
           B_end <= C_start.
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
        preselect=20,
        future_lambda=0.25,
        eps=1e-6,
    ):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.series_dim = int(series_dim)
        self.topk = int(topk)
        self.stride = int(stride)
        self.mode = int(mode)
        self.bins = int(bins)
        self.preselect = int(preselect)
        self.future_lambda = float(future_lambda)
        self.eps = float(eps)

        if self.mode not in (1, 2, 3):
            raise NotImplementedError(
                'HCTRetriever supports hct_mode=1, 2, or 3.'
            )
        if self.bins <= 0:
            raise ValueError('hct_bins must be > 0.')
        if self.preselect < self.topk:
            raise ValueError(
                'B3 preselect size must be >= hct_topk. '
                f'Got preselect={self.preselect}, topk={self.topk}.'
            )
        if self.future_lambda < 0:
            raise ValueError('hct_future_lambda must be >= 0.')

        self._memory = None
        self._input_dim = None
        self._exo_dim = None

        self._candidate_starts = None
        self._candidate_ends = None
        self._candidate_src = None
        self._candidate_fut = None

        self._candidate_endo_feat = None
        self._candidate_past_exo_feat = None
        self._candidate_exo_transition_feat = None

    # ------------------------------------------------------------------
    # Generic feature helpers
    # ------------------------------------------------------------------
    def _normalize_windows(self, x):
        """Per-window, per-channel z-normalization along time."""
        mean = x.mean(dim=-2, keepdim=True)
        std = x.std(dim=-2, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)

    def _window_features(self, x):
        """x: [N, T, D] -> L2-normalized flattened features [N, T*D]."""
        if x.ndim != 3:
            raise ValueError('HCT feature input must have shape [N, T, D].')
        if x.shape[-1] <= 0:
            raise ValueError('HCT feature input must contain at least one channel.')
        x = self._normalize_windows(x)
        x = x.reshape(x.shape[0], -1)
        return F.normalize(x, p=2, dim=-1)

    def _pool_time(self, x):
        """
        Adaptive-average-pool time to self.bins.

        x: [N, T, D]
        return: [N, bins, D]
        """
        if x.ndim != 3:
            raise ValueError('Pooling input must have shape [N, T, D].')
        x = x.transpose(1, 2)              # [N, D, T]
        x = F.adaptive_avg_pool1d(x, self.bins)
        return x.transpose(1, 2)           # [N, bins, D]

    def _exo_transition_features(self, src_exo, fut_exo):
        """
        Build exogenous transition features using SOURCE-window statistics.

        Historical:
            src_exo = A_exo, fut_exo = B_exo
        Current:
            src_exo = C_exo, fut_exo = D_exo

        Source-based normalization preserves the level/scale change from
        source to future instead of normalizing the future independently.

        The two windows may have different lengths (e.g. L=96, H=720):
        each is pooled to `bins`, then the pooled difference is flattened.
        """
        if src_exo.ndim != 3 or fut_exo.ndim != 3:
            raise ValueError(
                'Exogenous transition inputs must have shape [N, T, D_exo].'
            )
        if src_exo.shape[0] != fut_exo.shape[0]:
            raise ValueError('Source/future exogenous batch sizes do not match.')
        if src_exo.shape[-1] != fut_exo.shape[-1]:
            raise ValueError('Source/future exogenous channel counts do not match.')

        mean = src_exo.mean(dim=1, keepdim=True)
        std = src_exo.std(dim=1, keepdim=True, unbiased=False)

        src_n = (src_exo - mean) / (std + self.eps)
        fut_n = (fut_exo - mean) / (std + self.eps)

        src_p = self._pool_time(src_n)
        fut_p = self._pool_time(fut_n)

        transition = fut_p - src_p
        transition = transition.reshape(transition.shape[0], -1)
        return F.normalize(transition, p=2, dim=-1)

    def _zscore_1d(self, x):
        """Query-wise z-score over a candidate vector."""
        if x.numel() <= 1:
            return torch.zeros_like(x)
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + self.eps)

    def _validate_exog(self, input_dim):
        exo_dim = int(input_dim) - self.series_dim
        if self.mode in (2, 3) and exo_dim <= 0:
            raise ValueError(
                f'hct_mode={self.mode} requires exogenous variables.'
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

        Only A_endo and B_endo are later injected into the HCT network.
        """
        if not torch.is_tensor(full_series):
            full_series = torch.as_tensor(full_series, dtype=torch.float32)

        full_series = full_series.detach().cpu().float().contiguous()
        if full_series.ndim != 2:
            raise ValueError('full_series must have shape [N, D_all].')
        if full_series.shape[-1] < self.series_dim:
            raise ValueError('full_series has fewer channels than series_dim.')

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
            self._candidate_exo_transition_feat = None
            return

        starts = torch.arange(
            0, max_start + 1, self.stride, dtype=torch.long
        )

        src_endo = torch.stack([
            full_series[j:j + L, :self.series_dim]
            for j in starts.tolist()
        ], dim=0)
        fut_endo = torch.stack([
            full_series[j + L:j + L + H, :self.series_dim]
            for j in starts.tolist()
        ], dim=0)

        self._candidate_starts = starts
        self._candidate_ends = starts + L + H
        self._candidate_src = src_endo
        self._candidate_fut = fut_endo
        self._candidate_endo_feat = self._window_features(src_endo)

        if self.mode in (2, 3):
            src_exo = torch.stack([
                full_series[j:j + L, self.series_dim:]
                for j in starts.tolist()
            ], dim=0)
            self._candidate_past_exo_feat = self._window_features(src_exo)
        else:
            src_exo = None
            self._candidate_past_exo_feat = None

        if self.mode == 3:
            fut_exo = torch.stack([
                full_series[j + L:j + L + H, self.series_dim:]
                for j in starts.tolist()
            ], dim=0)
            self._candidate_exo_transition_feat = (
                self._exo_transition_features(src_exo, fut_exo)
            )
        else:
            self._candidate_exo_transition_feat = None

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
                'B3 / hct_mode=3 requires known future exogenous input '
                'future_exog=[B, pred_len, exo_dim].'
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
                'B3 future-exogenous shape mismatch. Expected '
                f'{expected}, received {tuple(future_exog.shape)}. '
                'Pass ONLY D_exo, never D_endo.'
            )
        return future_exog

    def _score_contexts(self, contexts, future_exog=None):
        """
        Return base/context scores and component score matrices [B, M].

        For B3 the returned `scores` are S_ctx (the exact B2 score), NOT the
        final B3 score. Final B3 re-ranking is performed after causal Top-M
        preselection inside `_select_topk_b3`.
        """
        if contexts.ndim != 3:
            raise ValueError('query contexts must have shape [B, L, D].')
        if contexts.shape[1] != self.seq_len:
            raise ValueError(
                f'HCT context length mismatch: {contexts.shape[1]} != {self.seq_len}.'
            )
        if contexts.shape[-1] < self.series_dim:
            raise ValueError('HCT query has fewer endogenous channels than series_dim.')

        query_endo = contexts[..., :self.series_dim]
        query_endo_feat = self._window_features(query_endo)
        score_endo = torch.matmul(
            query_endo_feat,
            self._candidate_endo_feat.transpose(0, 1),
        )
        components = {'endo': score_endo}

        if self.mode == 1:
            return score_endo, components

        query_exo = contexts[..., self.series_dim:]
        if query_exo.shape[-1] != self._exo_dim:
            raise ValueError(
                f'B{self.mode} query has {query_exo.shape[-1]} exogenous '
                f'channels, memory has {self._exo_dim}. Pass the FULL current '
                'history (Endo + Past Exo).'
            )

        query_past_exo_feat = self._window_features(query_exo)
        score_past_exo = torch.matmul(
            query_past_exo_feat,
            self._candidate_past_exo_feat.transpose(0, 1),
        )
        components['past_exo'] = score_past_exo

        score_context = 0.5 * (score_endo + score_past_exo)
        components['context'] = score_context

        if self.mode == 2:
            return score_context, components

        # B3 future-exogenous TRANSITION similarity.
        future_exog = self._prepare_future_exog(
            future_exog=future_exog,
            batch_size=contexts.shape[0],
        )
        query_transition_feat = self._exo_transition_features(
            query_exo,
            future_exog,
        )
        score_future_transition = torch.matmul(
            query_transition_feat,
            self._candidate_exo_transition_feat.transpose(0, 1),
        )
        components['future_exo'] = score_future_transition

        # Return B2 context score as the Stage-1 score.
        return score_context, components

    # ------------------------------------------------------------------
    # Output pack helpers
    # ------------------------------------------------------------------
    def _empty_pack(self, batch_size):
        B = int(batch_size)
        K = self.topk
        dtype = self._memory.dtype

        pack = {
            'hist_src': torch.zeros(
                B, K, self.seq_len, self.series_dim, dtype=dtype
            ),
            'hist_fut': torch.zeros(
                B, K, self.pred_len, self.series_dim, dtype=dtype
            ),
            'hist_score': torch.zeros(B, K, dtype=dtype),
            'hist_score_endo': torch.zeros(B, K, dtype=dtype),
            'hist_valid': torch.zeros(B, dtype=torch.bool),
            'hist_start': torch.full((B, K), -1, dtype=torch.long),
        }

        if self.mode in (2, 3):
            pack['hist_score_past_exo'] = torch.zeros(B, K, dtype=dtype)
            pack['hist_score_context'] = torch.zeros(B, K, dtype=dtype)

        if self.mode == 3:
            # Raw transition similarity and its z-score inside the preselection.
            pack['hist_score_future_exo'] = torch.zeros(B, K, dtype=dtype)
            pack['hist_score_context_z'] = torch.zeros(B, K, dtype=dtype)
            pack['hist_score_future_exo_z'] = torch.zeros(B, K, dtype=dtype)

        return pack

    def _fill_common_selected(
        self,
        pack,
        b,
        selected_idx,
        selected_final_scores,
        component_scores,
        context_z=None,
        future_z=None,
    ):
        K = self.topk
        keep = int(selected_idx.numel())
        if keep == 0:
            return

        # Pad by repeating the last valid candidate if fewer than K exist.
        if keep < K:
            pad_count = K - keep
            selected_idx = torch.cat([
                selected_idx,
                selected_idx[-1:].repeat(pad_count),
            ], dim=0)
            selected_final_scores = torch.cat([
                selected_final_scores,
                selected_final_scores[-1:].repeat(pad_count),
            ], dim=0)
            if context_z is not None:
                context_z = torch.cat([
                    context_z,
                    context_z[-1:].repeat(pad_count),
                ], dim=0)
            if future_z is not None:
                future_z = torch.cat([
                    future_z,
                    future_z[-1:].repeat(pad_count),
                ], dim=0)

        pack['hist_src'][b] = self._candidate_src[selected_idx]
        pack['hist_fut'][b] = self._candidate_fut[selected_idx]
        pack['hist_score'][b] = selected_final_scores
        pack['hist_start'][b] = self._candidate_starts[selected_idx]
        pack['hist_valid'][b] = True
        pack['hist_score_endo'][b] = component_scores['endo'][b, selected_idx]

        if self.mode in (2, 3):
            pack['hist_score_past_exo'][b] = (
                component_scores['past_exo'][b, selected_idx]
            )
            pack['hist_score_context'][b] = (
                component_scores['context'][b, selected_idx]
            )

        if self.mode == 3:
            pack['hist_score_future_exo'][b] = (
                component_scores['future_exo'][b, selected_idx]
            )
            pack['hist_score_context_z'][b] = context_z
            pack['hist_score_future_exo_z'][b] = future_z

    def _select_topk_standard(self, scores, component_scores, causal_mask):
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

            self._fill_common_selected(
                pack=pack,
                b=b,
                selected_idx=selected_idx,
                selected_final_scores=top_scores,
                component_scores=component_scores,
            )
        return pack

    def _select_topk_b3(self, context_scores, component_scores, causal_mask):
        """
        B3 two-stage selection.

        1) Causality filter.
        2) Top-M by exact B2 context score.
        3) z-score context and future-transition scores inside Top-M.
        4) final = z_context + lambda * z_future_transition.
        5) Top-K by final score.

        With lambda=0 and preselect>=topk, final Top-K is exactly the B2 Top-K
        (up to mathematically irrelevant score ties).
        """
        B = context_scores.shape[0]
        K = self.topk
        M = self.preselect
        pack = self._empty_pack(B)

        for b in range(B):
            valid_idx = torch.nonzero(causal_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue

            valid_ctx = context_scores[b, valid_idx]
            pre_keep = min(M, int(valid_idx.numel()))
            _, pre_local = torch.topk(
                valid_ctx, k=pre_keep, largest=True, sorted=True
            )
            pre_idx = valid_idx[pre_local]

            ctx_pre = component_scores['context'][b, pre_idx]
            fut_pre = component_scores['future_exo'][b, pre_idx]
            ctx_z_pre = self._zscore_1d(ctx_pre)
            fut_z_pre = self._zscore_1d(fut_pre)

            final_pre = ctx_z_pre + self.future_lambda * fut_z_pre

            keep = min(K, pre_keep)
            top_final, final_local = torch.topk(
                final_pre, k=keep, largest=True, sorted=True
            )
            selected_idx = pre_idx[final_local]
            selected_ctx_z = ctx_z_pre[final_local]
            selected_fut_z = fut_z_pre[final_local]

            self._fill_common_selected(
                pack=pack,
                b=b,
                selected_idx=selected_idx,
                selected_final_scores=top_final,
                component_scores=component_scores,
                context_z=selected_ctx_z,
                future_z=selected_fut_z,
            )

        return pack

    def _select_topk(self, scores, component_scores, causal_mask):
        if self.mode == 3:
            return self._select_topk_b3(
                context_scores=scores,
                component_scores=component_scores,
                causal_mask=causal_mask,
            )
        return self._select_topk_standard(
            scores=scores,
            component_scores=component_scores,
            causal_mask=causal_mask,
        )

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

        contexts = torch.zeros(
            B, L, self._input_dim, dtype=self._memory.dtype
        )
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
                self._candidate_ends.unsqueeze(0)
                <= query_starts.unsqueeze(1)
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
        """Retrieval for forecasting when the current C is supplied directly."""
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
                f'HCT forecast context length mismatch: '
                f'{query_contexts.shape[1]} != {self.seq_len}.'
            )

        if self.mode in (2, 3):
            if query_contexts.shape[-1] != self._input_dim:
                raise ValueError(
                    f'B{self.mode} forecast retrieval requires FULL current '
                    f'history. Expected {self._input_dim} channels, got '
                    f'{query_contexts.shape[-1]}.'
                )
        elif query_contexts.shape[-1] < self.series_dim:
            raise ValueError('Forecast context has too few endogenous channels.')

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
            pack['hist_src'][0],
            pack['hist_fut'][0],
            pack['hist_score'][0],
            bool(pack['hist_valid'][0].item()),
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
        if not bool(pack['hist_valid'][0].item()):
            return None

        items = []
        for rank in range(self.topk):
            start = int(pack['hist_start'][0, rank].item())
            if start < 0:
                continue

            item = {
                'start': start,
                'a_start': start,
                'a_end': start + self.seq_len,
                'b_start': start + self.seq_len,
                'b_end': start + self.seq_len + self.pred_len,
                'score': float(pack['hist_score'][0, rank].item()),
                'score_endo': float(pack['hist_score_endo'][0, rank].item()),
                'A_endo': pack['hist_src'][0, rank],
                'B_endo': pack['hist_fut'][0, rank],
            }

            if self.mode in (2, 3):
                item['score_past_exo'] = float(
                    pack['hist_score_past_exo'][0, rank].item()
                )
                item['score_context'] = float(
                    pack['hist_score_context'][0, rank].item()
                )

            if self.mode == 3:
                item['score_future_exo_transition'] = float(
                    pack['hist_score_future_exo'][0, rank].item()
                )
                item['score_context_z'] = float(
                    pack['hist_score_context_z'][0, rank].item()
                )
                item['score_future_exo_z'] = float(
                    pack['hist_score_future_exo_z'][0, rank].item()
                )

            items.append(item)
        return items
