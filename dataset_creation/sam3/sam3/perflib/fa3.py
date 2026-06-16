# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
import torch.nn.functional as F


def _is_hopper_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9  # Hopper is SM90+


_USE_FA3 = _is_hopper_gpu()


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


def flash_attn_func(q, k, v):
    if _USE_FA3:
        return flash_attn_func_op(q, k, v)
    # q/k/v arrive as [B, seq_len, num_heads, head_dim] (FA3 layout).
    # SDPA expects [B, num_heads, seq_len, head_dim], so transpose in/out.
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
