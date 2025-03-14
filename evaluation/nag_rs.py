import torch

'''
NAG (with option to restart)
Effectively the one on WCRR repo https://github.com/axgoujon/weakly_convex_ridge_regularizer/blob/main/models/utils.py
Also see Sec 3.2 https://arxiv.org/pdf/1204.3982
'''

def reconstruct_NAG_RS(
    y,
    physics,
    data_fidelity,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    detach_grads=False,
    verbose=False,
    x_init=None,
    progress=False,
    restart=False
):
    
    if progress & (y.shape[0]!=1):
        progress = False
        print("Progress not supported for batch processing")

    # run Nesterov Accelerated Gradient
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)
    z = torch.clone(x)
    t = torch.ones(x.shape[0], device=x.device).view(-1,1,1,1)
    
    # Def func and grad
    def f(x_in):
        return data_fidelity(x_in, y, physics) + lmbd * regularizer.g(x_in)
    
    def grad_f(x_in,detach_grads):
        grad = data_fidelity.grad(x_in, y, physics).detach() + lmbd * regularizer.grad(x_in)
        return grad.detach() if detach_grads else grad
            
    # the index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    # relative change in the estimate
    res = (NAG_tol + 1)*torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    if progress:
        energy_list = [f(x).item()]
        grad_norm_list = [torch.norm(grad_f(x,True)).item()]
        res_list = []

    for step in range(NAG_max_iter):
        x_old = torch.clone(x)
        grad = grad_f(z,detach_grads)[idx]
        x[idx] = z[idx] - NAG_step_size * grad
        #x = x.clamp(0, 1)
        t_old = torch.clone(t)
        t = 0.5*(1 + torch.sqrt(1 + 4 * t**2))
        z[idx] = x[idx] + (t_old[idx] - 1)/t[idx] * (x[idx] - x_old[idx])

        if step > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1,2,3)) / (torch.norm(x[idx], p=2, dim=(1,2,3)))
        
        if restart:
            esti = torch.sum(grad*(x[idx] - x_old[idx]), dim=(1,2,3))
            id_restart = (esti > 0).nonzero().view(-1)
            t[idx[id_restart]] = 1
            z[idx[id_restart]] = x[idx[id_restart]]

        condition = (res > NAG_tol)
        idx = condition.nonzero().view(-1)

        if progress:
            energy_list.append(f(x).item())
            grad_norm_list.append(torch.norm(grad_f(x,True)).item())
            res_list.append(res.item())
        
        if torch.max(res) < NAG_tol:
            if verbose:
                print(f"Converged in iter {step}, tol {torch.max(res).item():.6f}")
            break
            
    if verbose and (torch.max(res) >= NAG_tol):
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    
    return (x, energy_list, grad_norm_list, res_list) if progress else x
