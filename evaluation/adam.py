import torch 
from tqdm import tqdm 
import torch.nn.functional as F

def reconstruct_adam(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size,
    max_iter,
    tol,
    x_init=None,
    detach_grads=True,
    verbose=False,
    return_stats=False,
    psnr_fun = None, 
):
    """wrapper for adam"""




    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y) 

    # Remove 2-pixel boundary
    x = x[:, :, 2:-2, 2:-2]
    # Add replicate padding (2-pixel boundary restored with replicate padding)
    x = F.pad(x, (2, 2, 2, 2), mode='replicate')
    #import matplotlib.pyplot as plt 
    #plt.figure()
    #plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
    #plt.show()

    def energy(val, y_in):
        fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)

        return fun.reshape(-1)

    
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=step_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=step_size/10.)
    for i in (progress_bar := tqdm(range(max_iter))):
        optimizer.zero_grad()
        loss = energy(x, y)     
        loss.backward()
        optimizer.step()
        if not psnr_fun is None:
            psnr_ = psnr_fun(x)
            progress_bar.set_description(
                "Step {} || Loss {:.3f} || PSNR {:.3f}".format(
                    i + 1, loss.item(), psnr_
                )
            )
        else:
            progress_bar.set_description(
                "Step {} || Loss {} ".format(
                    i + 1, loss.item()
                )
            )
        scheduler.step() 
        with torch.no_grad():
            x.data.clamp_(min=0)

        #import matplotlib.pyplot as plt 
        #plt.figure()
        #plt.imshow(x[0,0].detach().cpu().numpy(), cmap="gray")
        #plt.show()

    del optimizer 
    del scheduler

    rec = x.detach()
    if return_stats:
        return rec, None 
    return rec