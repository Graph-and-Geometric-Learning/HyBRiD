import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Masker(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_nodes: int,
    ) -> None:
        super().__init__()

        self.mask = nn.Parameter(torch.Tensor(n_heads, n_nodes, 2))
        nn.init.xavier_normal_(self.mask)

    def forward(self) -> tuple[Tensor, Tensor]:
        """
        Outputs:
            mask - [n_heads (n_hypers), n_nodes], node selection
            mask_logits - [n_heads (n_hypers), n_nodes], logits of node selection probability
        """
        mask_logits = self.mask
        mask_logits = torch.log(mask_logits.softmax(-1))

        mask = F.gumbel_softmax(mask_logits, tau=1, hard=True)[..., 1]

        return mask, mask_logits[..., 1]


class HyBRiDConstructor(nn.Module):
    def __init__(self, n_hypers: int, n_nodes: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.mask = Masker(n_hypers, n_nodes)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Inputs:
            x - [batch_size, n_nodes, feature_dim], note that feature_dim = n_nodes
        Outputs:
            h - [batch_size, n_hypers, feature_dim]
            mask - [n_hypers, n_nodes], node selection
            mask_logits - [n_hypers, n_nodes], logits of node selection probability
        """
        bs, n_nodes, dim = x.size()
        mask, mask_logits = self.mask()

        x = x[:, None, :, :] * mask[None, :, :, None]
        h = x.sum(-2) / (1e-7 + mask.sum(-1)[None, :, None])

        h = self.dropout(h)

        return h, (mask, mask_logits)


class HyBRiDWeighter(nn.Module):
    def __init__(
        self, d_model: int, hidden_size: int, n_hypers: int, layer_norm_eps=1e-5
    ) -> None:
        super().__init__()
        self.dim_reduction = nn.Sequential(
            nn.LayerNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model),
            nn.LayerNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, 1),
        )
        self.last = nn.Linear(n_hypers, 1)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inputs:
            h - [batch_size, n_hypers, feature_dim]
        Outputs:
            preds - [batch_size], predicted targets (e.g. IQ)
            last - [batch_size, n_hypers], used as weights of hyperedges
        """
        bs = h.size(0)
        h = self.dim_reduction(h)
        last = h.reshape((bs, -1))
        preds = self.last(last).squeeze()

        return preds, last


class HyBRiD(nn.Module):
    def __init__(self, n_hypers: int, hidden_size: int, n_nodes: int, dropout: float) -> None:
        super().__init__()
        self.constructor = HyBRiDConstructor(n_hypers=n_hypers, n_nodes=n_nodes, dropout=dropout)
        self.weighter = HyBRiDWeighter(
            d_model=n_nodes, hidden_size=hidden_size, n_hypers=n_hypers
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Inputs:
            x - [batch_size, n_nodes, feature_dim], note that feature_dim = n_nodes
        Outputs:
            preds - [batch_size], predicted targets (e.g. IQ)
            last - [batch_size, n_hypers], used as weights of hyperedges
            mask - [n_hypers, n_nodes], node selection
            mask_logits - [n_hypers, n_nodes], logits of node selection probability
        """

        h, (mask, mask_logits) = self.constructor(x)
        preds, last = self.weighter(h)

        return {
            "preds": preds,
            "last": last.detach(),
            "mask": mask,
            "mask_logits": mask_logits,
        }
