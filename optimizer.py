from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()
                # # Adam with L2 regularization (not used in this assignment)
                # grad = grad + group['weight_decay'] * p.data

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    # at t=0, initialize state
                    state['t'] = 0
                    state['avg'] = torch.zeros_like(p, device=p.device) # m_t
                    state['avg_sq'] = torch.zeros_like(p, device=p.device) # v_t
                t = state['t'] + 1
                m_t, v_t = state['avg'], state['avg_sq']
                beta1, beta2 = group['betas'][0], group['betas'][1]

                # Access hyperparameters from the `group` dictionary
                alpha = group['lr']

                # Update first and second moments of the gradients
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * grad.square()
                state['t'] = t
                state['avg'] = m_t
                state['avg_sq'] = v_t

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                # "efficient" method of computing the bias correction as in Section 2 of Adam paper
                if group['correct_bias']:
                    beta1_t, beta2_t = beta1 ** t, beta2 ** t
                    alpha_t = alpha * (torch.tensor([1 - beta2_t], device=p.device).sqrt() / (1 - beta1_t))
                else:
                    alpha_t = alpha

                # Update parameters
                p_t = p - alpha_t * m_t / (torch.sqrt(v_t) + group['eps'])

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p_t = p_t - alpha * group['weight_decay'] * p
                p.data = p_t

        return loss