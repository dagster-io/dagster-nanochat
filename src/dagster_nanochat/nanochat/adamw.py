"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""

import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """

    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                if grad is None:
                    # Skip parameters with no gradient (e.g., unused vocab extensions)
                    continue
                dim0 = grad.shape[0]
                # Handle tensors not evenly divisible by world_size
                rank_size = (dim0 + world_size - 1) // world_size  # Round up
                padded_size = rank_size * world_size

                # Pad gradient if needed
                if padded_size > dim0:
                    padding = padded_size - dim0
                    grad_padded = torch.nn.functional.pad(
                        grad, (0, 0) * (grad.ndim - 1) + (0, padding)
                    )
                else:
                    grad_padded = grad

                grad_slice = torch.empty_like(grad_padded[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(
                        grad_slice, grad_padded, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future()
                )
                grad_slices.append((grad_slice, dim0))

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                dim0 = p.shape[0]
                rank_size = (dim0 + world_size - 1) // world_size

                # Calculate slice boundaries for this rank
                start_idx = rank * rank_size
                end_idx = min((rank + 1) * rank_size, dim0)
                actual_slice_size = end_idx - start_idx

                p_slice = p[start_idx:end_idx]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice, orig_dim0 = grad_slices[idx]
                # Trim padding if this rank has it
                if actual_slice_size < rank_size:
                    g_slice = g_slice[:actual_slice_size]
                # State init
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1

                # Pad p_slice for all_gather if needed
                padded_size = rank_size * world_size
                if padded_size > dim0:
                    p_slice_padded = torch.nn.functional.pad(
                        p_slice,
                        (0, 0) * (p.ndim - 1) + (0, rank_size - actual_slice_size),
                    )
                    p_gathered = torch.empty(
                        (padded_size,) + p.shape[1:], dtype=p.dtype, device=p.device
                    )
                    future = dist.all_gather_into_tensor(
                        p_gathered, p_slice_padded, async_op=True
                    ).get_future()
                    all_reduce_futures.append((future, p, p_gathered, dim0))
                else:
                    all_reduce_futures.append(
                        (
                            dist.all_gather_into_tensor(
                                p, p_slice, async_op=True
                            ).get_future(),
                            None,
                            None,
                            None,
                        )
                    )

        # Wait for all futures and unpad if necessary
        for item in all_reduce_futures:
            future, p_orig, p_gathered, orig_dim = item
            future.wait()
            if p_orig is not None:
                # Copy back the unpadded portion from gathered tensor
                p_orig.copy_(p_gathered[:orig_dim])
