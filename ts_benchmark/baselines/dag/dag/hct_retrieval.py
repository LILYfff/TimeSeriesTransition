import torch
import torch.nn.functional as F


class HCTRetriever:
    """
    Historical Conditional Transition Retriever.

    Current formal stage:
        mode = 1 -> Endo-only retrieval

    For current context C and a historical transition A -> B:
        score = cosine(normalize(A_endo), normalize(C_endo))

    Strict causal rule for index-based training/validation retrieval:
        B_end <= C_start
        j + seq_len + pred_len <= query_start

    The retriever keeps a vectorized candidate bank so the same historical
    windows are not re-normalized on every batch and every epoch.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        series_dim,
        topk=5,
        stride=12,
        mode=1,
        eps=1e-6,
    ):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.series_dim = int(series_dim)
        self.topk = int(topk)
        self.stride = int(stride)
        self.mode = int(mode)
        self.eps = float(eps)

        if self.mode != 1:
            raise NotImplementedError(
                "This formal HCT stage implements B1 / hct_mode=1 "
                "(Endo-only retrieval). B2/B3 retrieval will be added "
                "separately so their exogenous contexts are not silently "
                "treated as B1."
            )

        self._memory = None
        self._candidate_starts = None
        self._candidate_ends = None
        self._candidate_src = None
        self._candidate_fut = None
        self._candidate_feat = None

    def _normalize_windows(self, x):
        """
        x: [..., T, D]
        Normalize each channel along the temporal dimension.
        """
        mean = x.mean(dim=-2, keepdim=True)
        std = x.std(
            dim=-2,
            keepdim=True,
            unbiased=False,
        )
        return (x - mean) / (std + self.eps)

    def _window_features(self, x):
        """
        x: [B, T, D] or [M, T, D]
        -> [B/M, T*D], L2-normalized.
        """
        x = self._normalize_windows(x)
        x = x.reshape(x.shape[0], -1)
        return F.normalize(
            x,
            p=2,
            dim=-1,
        )

    def prepare_memory(self, full_series):
        """
        Prepare a frozen candidate bank.

        full_series:
            [N, D_all], CPU tensor preferred.

        Candidate windows are:
            A = [j, j+L)
            B = [j+L, j+L+H)
        for j = 0, stride, 2*stride, ...
        """
        if not torch.is_tensor(full_series):
            full_series = torch.as_tensor(
                full_series,
                dtype=torch.float32,
            )

        full_series = full_series.detach().cpu().float().contiguous()
        self._memory = full_series

        n = full_series.shape[0]
        L = self.seq_len
        H = self.pred_len

        max_start = n - L - H

        if max_start < 0:
            self._candidate_starts = torch.empty(
                0,
                dtype=torch.long,
            )
            self._candidate_ends = torch.empty(
                0,
                dtype=torch.long,
            )
            self._candidate_src = torch.empty(
                0,
                L,
                self.series_dim,
                dtype=full_series.dtype,
            )
            self._candidate_fut = torch.empty(
                0,
                H,
                self.series_dim,
                dtype=full_series.dtype,
            )
            self._candidate_feat = torch.empty(
                0,
                L * self.series_dim,
                dtype=full_series.dtype,
            )
            return

        starts = torch.arange(
            0,
            max_start + 1,
            self.stride,
            dtype=torch.long,
        )

        src = torch.stack(
            [
                full_series[
                    j : j + L,
                    : self.series_dim,
                ]
                for j in starts.tolist()
            ],
            dim=0,
        )

        fut = torch.stack(
            [
                full_series[
                    j + L : j + L + H,
                    : self.series_dim,
                ]
                for j in starts.tolist()
            ],
            dim=0,
        )

        self._candidate_starts = starts
        self._candidate_ends = starts + L + H
        self._candidate_src = src
        self._candidate_fut = fut
        self._candidate_feat = self._window_features(src)

    def _ensure_memory(self, full_series):
        if self._memory is None:
            self.prepare_memory(full_series)
            return

        if torch.is_tensor(full_series):
            same_shape = tuple(full_series.shape) == tuple(self._memory.shape)
        else:
            same_shape = tuple(full_series.shape) == tuple(self._memory.shape)

        if not same_shape:
            self.prepare_memory(full_series)

    def _select_topk(
        self,
        query_feat,
        causal_mask,
    ):
        """
        query_feat:
            [B, F]

        causal_mask:
            [B, M], True for allowed candidates.

        Returns:
            hist_src   [B, K, L, D]
            hist_fut   [B, K, H, D]
            hist_score [B, K]
            hist_valid [B]
            hist_start [B, K]
        """
        B = query_feat.shape[0]
        K = self.topk
        dtype = self._memory.dtype

        out_src = torch.zeros(
            B,
            K,
            self.seq_len,
            self.series_dim,
            dtype=dtype,
        )
        out_fut = torch.zeros(
            B,
            K,
            self.pred_len,
            self.series_dim,
            dtype=dtype,
        )
        out_score = torch.zeros(
            B,
            K,
            dtype=dtype,
        )
        out_valid = torch.zeros(
            B,
            dtype=torch.bool,
        )
        out_start = torch.full(
            (B, K),
            -1,
            dtype=torch.long,
        )

        if self._candidate_feat.shape[0] == 0:
            return {
                "hist_src": out_src,
                "hist_fut": out_fut,
                "hist_score": out_score,
                "hist_valid": out_valid,
                "hist_start": out_start,
            }

        scores = torch.matmul(
            query_feat,
            self._candidate_feat.transpose(0, 1),
        )  # [B, M]

        for b in range(B):
            valid_idx = torch.nonzero(
                causal_mask[b],
                as_tuple=False,
            ).flatten()

            if valid_idx.numel() == 0:
                continue

            valid_scores = scores[b, valid_idx]
            keep = min(
                K,
                valid_idx.numel(),
            )

            top_scores, local_idx = torch.topk(
                valid_scores,
                k=keep,
                largest=True,
                sorted=True,
            )
            selected_idx = valid_idx[local_idx]

            # If fewer than K causal candidates exist, repeat the last one.
            if keep < K:
                pad_count = K - keep
                selected_idx = torch.cat(
                    [
                        selected_idx,
                        selected_idx[-1:].repeat(pad_count),
                    ],
                    dim=0,
                )
                top_scores = torch.cat(
                    [
                        top_scores,
                        top_scores[-1:].repeat(pad_count),
                    ],
                    dim=0,
                )

            out_src[b] = self._candidate_src[selected_idx]
            out_fut[b] = self._candidate_fut[selected_idx]
            out_score[b] = top_scores
            out_start[b] = self._candidate_starts[selected_idx]
            out_valid[b] = True

        return {
            "hist_src": out_src,
            "hist_fut": out_fut,
            "hist_score": out_score,
            "hist_valid": out_valid,
            "hist_start": out_start,
        }

    def retrieve_batch(
        self,
        full_series,
        query_starts,
        max_b_end=None,
    ):
        """
        Vectorized index-based retrieval for training/validation.

        query_starts:
            [B], start index of current C in full_series.

        max_b_end:
            Optional global cutoff. If provided, every retrieved historical
            B must also satisfy B_end <= max_b_end. This is used for
            validation so held-out validation targets never become retrieval
            memory for later validation windows.

        Returns:
            hist_src   [B, K, L, D]
            hist_fut   [B, K, H, D]
            hist_score [B, K]
            hist_valid [B]
            hist_start [B, K]
        """
        self._ensure_memory(full_series)

        if torch.is_tensor(query_starts):
            query_starts = query_starts.detach().cpu().long()
        else:
            query_starts = torch.as_tensor(
                query_starts,
                dtype=torch.long,
            )

        query_starts = query_starts.reshape(-1)
        B = query_starts.shape[0]
        N = self._memory.shape[0]
        L = self.seq_len

        contexts = torch.zeros(
            B,
            L,
            self.series_dim,
            dtype=self._memory.dtype,
        )

        query_ok = (
            (query_starts >= 0)
            & (query_starts + L <= N)
        )

        for b, q in enumerate(query_starts.tolist()):
            if query_ok[b]:
                contexts[b] = self._memory[
                    q : q + L,
                    : self.series_dim,
                ]

        query_feat = self._window_features(contexts)

        if self._candidate_ends.numel() == 0:
            causal_mask = torch.zeros(
                B,
                0,
                dtype=torch.bool,
            )
        else:
            causal_mask = (
                self._candidate_ends.unsqueeze(0)
                <= query_starts.unsqueeze(1)
            )

            if max_b_end is not None:
                cutoff_mask = (
                    self._candidate_ends.unsqueeze(0)
                    <= int(max_b_end)
                )
                causal_mask = causal_mask & cutoff_mask

            causal_mask = (
                causal_mask
                & query_ok.unsqueeze(1)
            )

        return self._select_topk(
            query_feat=query_feat,
            causal_mask=causal_mask,
        )

    def retrieve_context_batch(
        self,
        full_series,
        query_contexts,
        max_b_end,
    ):
        """
        Retrieval for forecasting when current C is supplied directly.

        The candidate memory remains frozen. max_b_end defines the latest
        allowed end of historical B. For rolling evaluation we use the start
        of the earliest test context, so every retrieved transition remains
        strictly before the current context.

        query_contexts:
            [B, L, D_current]

        max_b_end:
            scalar integer. Historical candidates must satisfy B_end <= this.
        """
        self._ensure_memory(full_series)

        if not torch.is_tensor(query_contexts):
            query_contexts = torch.as_tensor(
                query_contexts,
                dtype=self._memory.dtype,
            )

        query_contexts = (
            query_contexts.detach().cpu().float()
        )

        if query_contexts.ndim == 2:
            query_contexts = query_contexts.unsqueeze(0)

        if query_contexts.shape[1] != self.seq_len:
            raise ValueError(
                "HCT forecast context length mismatch: "
                f"{query_contexts.shape[1]} != {self.seq_len}."
            )

        if query_contexts.shape[-1] < self.series_dim:
            raise ValueError(
                "HCT forecast context has fewer endogenous channels "
                "than series_dim."
            )

        contexts = query_contexts[
            ...,
            : self.series_dim,
        ]
        query_feat = self._window_features(contexts)

        B = contexts.shape[0]

        if self._candidate_ends.numel() == 0:
            causal_mask = torch.zeros(
                B,
                0,
                dtype=torch.bool,
            )
        else:
            allowed = (
                self._candidate_ends
                <= int(max_b_end)
            )
            causal_mask = allowed.unsqueeze(0).expand(
                B,
                -1,
            )

        return self._select_topk(
            query_feat=query_feat,
            causal_mask=causal_mask,
        )

    # ------------------------------------------------------------------
    # Compatibility/debug APIs retained from the previous development
    # stage.
    # ------------------------------------------------------------------
    def retrieve_tensors(
        self,
        full_series,
        query_start,
    ):
        pack = self.retrieve_batch(
            full_series=full_series,
            query_starts=[query_start],
        )

        return (
            pack["hist_src"][0],
            pack["hist_fut"][0],
            pack["hist_score"][0],
            bool(pack["hist_valid"][0].item()),
        )

    def retrieve(
        self,
        full_series,
        query_start,
    ):
        pack = self.retrieve_batch(
            full_series=full_series,
            query_starts=[query_start],
        )

        if not bool(pack["hist_valid"][0].item()):
            return None

        items = []

        for rank in range(self.topk):
            start = int(
                pack["hist_start"][0, rank].item()
            )

            if start < 0:
                continue

            items.append(
                {
                    "start": start,
                    "a_start": start,
                    "a_end": start + self.seq_len,
                    "b_start": start + self.seq_len,
                    "b_end": (
                        start
                        + self.seq_len
                        + self.pred_len
                    ),
                    "score": float(
                        pack["hist_score"][0, rank].item()
                    ),
                    "A_endo": pack["hist_src"][0, rank],
                    "B_endo": pack["hist_fut"][0, rank],
                }
            )

        return items if len(items) > 0 else None
