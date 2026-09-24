import torch
import torch.nn.functional as F


class HCTRetriever:
    """
    Historical Conditional Transition Retriever

    mode = 1:
        use endogenous history only

        current:
            C_endo

        historical candidate:
            A_endo -> B_endo
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        series_dim,
        topk=5,
        stride=12,
        eps=1e-6,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        self.topk = topk
        self.stride = stride
        self.eps = eps

    def _normalize(self, x):
        """
        x: [T, D]

        Normalize each channel along time.
        """
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(
            dim=0,
            keepdim=True,
            unbiased=False
        )

        return (x - mean) / (std + self.eps)

    def _flatten_normalize(self, x):
        """
        x: [T, D]
        -> normalized 1-D feature vector
        """
        x = self._normalize(x)
        x = x.reshape(-1)

        x = F.normalize(
            x,
            p=2,
            dim=0,
        )

        return x

    def _similarity(self, a, c):
        """
        Cosine similarity between
        historical A and current C.
        """
        a_vec = self._flatten_normalize(a)
        c_vec = self._flatten_normalize(c)

        return torch.dot(a_vec, c_vec)

    def retrieve(
        self,
        full_series,
        query_start,
    ):
        """
        full_series:
            [N, D_all]

        query_start:
            starting position of current C

        Requirement:
            historical B must finish
            before current C starts.

            j + seq_len + pred_len <= query_start
        """

        L = self.seq_len
        H = self.pred_len

        # current endogenous context C
        C_endo = full_series[
            query_start : query_start + L,
            : self.series_dim,
        ]

        candidates = []

        # Strict causal retrieval
        max_start = query_start - L - H

        if max_start < 0:
            return None

        for j in range(
            0,
            max_start + 1,
            self.stride,
        ):
            A_endo = full_series[
                j : j + L,
                : self.series_dim,
            ]

            B_endo = full_series[
                j + L : j + L + H,
                : self.series_dim,
            ]

            if (
                len(A_endo) != L
                or len(B_endo) != H
            ):
                continue

            score = self._similarity(
                A_endo,
                C_endo,
            )

            candidates.append(
                {
                    "start": j,
                    "a_start": j,
                    "a_end": j + L,
                    "b_start": j + L,
                    "b_end": j + L + H,
                    "score": score.item(),
                    "A_endo": A_endo,
                    "B_endo": B_endo,
                }
            )

        if len(candidates) == 0:
            return None

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return candidates[: self.topk]


    def retrieve_tensors(
        self,
        full_series,
        query_start,
    ):
        """
        Retrieve Top-K historical endogenous transitions.

        Returns
        -------
        hist_src:
            [K, L, series_dim]

        hist_fut:
            [K, H, series_dim]

        hist_score:
            [K]

        valid:
            bool
        """

        retrieved = self.retrieve(
            full_series=full_series,
            query_start=query_start,
        )

        # --------------------------------------------------------
        # No valid historical transition
        # --------------------------------------------------------
        if retrieved is None or len(retrieved) == 0:

            hist_src = torch.zeros(
                self.topk,
                self.seq_len,
                self.series_dim,
                dtype=full_series.dtype,
            )

            hist_fut = torch.zeros(
                self.topk,
                self.pred_len,
                self.series_dim,
                dtype=full_series.dtype,
            )

            hist_score = torch.zeros(
                self.topk,
                dtype=full_series.dtype,
            )

            return (
                hist_src,
                hist_fut,
                hist_score,
                False,
            )

        # --------------------------------------------------------
        # Convert retrieved transitions to tensors
        # --------------------------------------------------------
        src_list = []
        fut_list = []
        score_list = []

        for item in retrieved:

            src_list.append(
                item["A_endo"]
            )

            fut_list.append(
                item["B_endo"]
            )

            score_list.append(
                item["score"]
            )

        # --------------------------------------------------------
        # If fewer than K candidates exist,
        # repeat the last valid transition.
        # --------------------------------------------------------
        while len(src_list) < self.topk:

            src_list.append(
                src_list[-1].clone()
            )

            fut_list.append(
                fut_list[-1].clone()
            )

            score_list.append(
                score_list[-1]
            )

        hist_src = torch.stack(
            src_list[:self.topk],
            dim=0,
        )

        hist_fut = torch.stack(
            fut_list[:self.topk],
            dim=0,
        )

        hist_score = torch.tensor(
            score_list[:self.topk],
            dtype=full_series.dtype,
        )

        return (
            hist_src,
            hist_fut,
            hist_score,
            True,
        )

    def retrieve_batch(
        self,
        full_series,
        query_starts,
    ):
        """
        Retrieve historical transitions for a whole batch.

        Parameters
        ----------
        full_series:
            Tensor [N, D_all]

        query_starts:
            Tensor/List [B]

        Returns
        -------
        hct_pack:
            {
                "hist_src":   [B, K, L, D_endo],
                "hist_fut":   [B, K, H, D_endo],
                "hist_score": [B, K],
                "hist_valid": [B]
            }
        """

        batch_src = []
        batch_fut = []
        batch_score = []
        batch_valid = []

        for q in query_starts:

            if torch.is_tensor(q):
                query_start = int(q.item())
            else:
                query_start = int(q)

            (
                hist_src,
                hist_fut,
                hist_score,
                hist_valid,
            ) = self.retrieve_tensors(
                full_series=full_series,
                query_start=query_start,
            )

            batch_src.append(hist_src)
            batch_fut.append(hist_fut)
            batch_score.append(hist_score)
            batch_valid.append(hist_valid)

        hist_src = torch.stack(
            batch_src,
            dim=0,
        )

        hist_fut = torch.stack(
            batch_fut,
            dim=0,
        )

        hist_score = torch.stack(
            batch_score,
            dim=0,
        )

        hist_valid = torch.tensor(
            batch_valid,
            dtype=torch.bool,
        )

        return {
            "hist_src": hist_src,
            "hist_fut": hist_fut,
            "hist_score": hist_score,
            "hist_valid": hist_valid,
        }