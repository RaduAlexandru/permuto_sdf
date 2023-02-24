import torch

class VectorAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        super(VectorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)