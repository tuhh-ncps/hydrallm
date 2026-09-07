import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def linear_sliced(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_rows: Optional[int] = None,
    in_cols: Optional[int] = None,
) -> torch.Tensor:
    if in_cols is not None and input.size(-1) > in_cols:
        input = input[..., :in_cols]
    w = weight
    b = bias
    if out_rows is not None:
        w = w[:out_rows, :]
        if b is not None:
            b = b[:out_rows]
    if in_cols is not None:
        w = w[:, :in_cols]
    return F.linear(input, w, b)


class DynamicLinear(nn.Linear):
    def forward_sliced(
        self,
        input: torch.Tensor,
        out_rows: Optional[int] = None,
        in_cols: Optional[int] = None,
    ) -> torch.Tensor:
        return linear_sliced(input, self.weight, self.bias, out_rows, in_cols)


class DynamicTextScaledWordEmbedding(nn.Embedding):
    """
    Dynamic version of (Gemma3's) embedding, allowing output slicing.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(
            embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor, out_cols: Optional[int] = None) -> torch.Tensor:
        # The original forward is what we want to slice
        embeddings = F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Slice the output embedding dimension
        if out_cols is not None and embeddings.size(-1) > out_cols:
            embeddings = embeddings[..., :out_cols]

        # Apply the scale
        return embeddings * self.embed_scale.to(self.weight.dtype)

class DynamicRMSNorm(nn.Module):
    """
    Dynamic RMSNorm that can operate on sliced tensors.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, in_cols: Optional[int] = None) -> torch.Tensor:
        in_cols = in_cols or x.size(-1)

        # If input tensor is already sliced, we don't need to slice it again
        if x.size(-1) < self.weight.size(0):
            x_sliced = x
        else:
            x_sliced = x[..., :in_cols].contiguous()

        weight_sliced = self.weight[:in_cols]

        output = self._norm(x_sliced.float())
        output = output * (1.0 + weight_sliced.float())
        return output.type_as(x)
