from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1) # (bs, seqlen, n_local_heads, head_dim//2)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim)) # (head_dim//2,)
    t = torch.arange(seqlen, device=device)  # type: ignore # (seqlen,)
    freqs = torch.outer(t, freqs).float()  # type: ignore # (head_dim//2, seqlen)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos, sin = reshape_for_broadcast(cos, query_real), reshape_for_broadcast(sin, query_real)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_out_11, query_out_12 = query_real * cos, query_imag * cos
    query_out_21, query_out_22 = -query_imag * sin, query_real * sin
    query_out_1 = query_out_11+query_out_21
    query_out_2 = query_out_12+query_out_22

    key_out_11, key_out_12 = key_real * cos, key_imag * cos
    key_out_21, key_out_22 = -key_imag * sin, key_real * sin
    key_out_1 = key_out_11+key_out_21
    key_out_2 = key_out_12+key_out_22

    # concat query_out_1 and query_out_2 with alternate order on the last dimension
    query_out = torch.zeros_like(query, device=device)
    query_out[:, :, :, ::2] = query_out_1
    query_out[:, :, :, 1::2] = query_out_2
    
    key_out = torch.zeros_like(key, device=device)
    key_out[:, :, :, ::2] = key_out_1
    key_out[:, :, :, 1::2] = key_out_2
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out