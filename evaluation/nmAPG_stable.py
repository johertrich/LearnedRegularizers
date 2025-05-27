import torch
import numpy as np
from typing import Callable, Tuple, Union  # Added Tuple, Union for type hint


def nmAPG_stable(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],  # Modified f signature based on usage
    nabla: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],  # Modified nabla signature
    max_iter: int = 200,
    L_init: float = 1,
    tol: float = 1e-4,
    rho: float = 0.25,
    delta: float = 0.1,
    eta: float = 0.8,
    verbose: bool = False,
    return_L: bool = False,
) -> Union[
    Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]
]:  # Adjusted return hint
    x = x0.clone()
    x_old = x.clone()
    z = x0.clone()
    t = 1.0
    t_old = 0.0
    q = 1.0
    try:
        c = f(x, y)  # Initial energy calculation
    except Exception as e:
        print(f"Error calculating initial 'c': {e}. Check inputs/functions.")
        raise e

    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(
        0, x.shape[0], device=x.device
    )  # Initially all indices are active
    grad = torch.zeros_like(x)
    grad_old = torch.zeros_like(x)  # Initialize grad_old
    x_bar = torch.zeros_like(x)
    x_bar_old = torch.zeros_like(x)  # Initialize x_bar_old
    iterations = max_iter  # Default if loop finishes by max_iter
    if torch.isnan(x).any():
        if verbose:
            print("NaN detected in x, returning Inf.")
        return torch.full_like(x, float("inf")), iterations if not return_L else L
    for i in range(max_iter):
        # --- Extrapolation Step ---
        current_x_old = x.clone()  # Store x before any updates in this iteration

        x_bar[idx] = (
            x[idx]
            + t_old / t * (z[idx] - x[idx])
            + (t_old - 1)
            / t
            * (x[idx] - x_old[idx])  # Uses x_old from start of *previous* iter
        )
        # x_old needs to store the state *before* x is updated in this iteration

        # --- Gradient Calculation ---
        try:
            energy, grad[idx] = f(x_bar[idx], y[idx]), nabla(x_bar[idx], y[idx])
        except Exception as e:
            print(f"Error in f() or nabla() at iter {i} for idx {idx}: {e}")
            # Option: return previous state or raise
            # Returning previous state might be safer in a long run
            iterations = i
            if return_L:
                return x_old, L  # Return L corresponding to x_old
            else:
                return x_old, iterations

        # --- Lipschitz Update (Barzilai-Borwein style step) ---
        if i > 0:
            dx_grad = grad[idx] - grad_old[idx]
            dx_iter = x_bar[idx] - x_bar_old[idx]
            s = (dx_grad * dx_grad).sum((1, 2, 3), keepdim=True)
            denom = (dx_grad * dx_iter).sum((1, 2, 3), keepdim=True)
            # Add safety for denominator
            safe_denom_mask_4d = torch.abs(denom) > 1e-8
            safe_denom_mask_1d = safe_denom_mask_4d.squeeze()

            if safe_denom_mask_1d.any():
                indices_to_update = idx[safe_denom_mask_1d]
                s_safe = s[safe_denom_mask_1d]
                denom_safe = denom[safe_denom_mask_1d]
                L_update = s_safe / denom_safe
                # Clip L update, add max value for stability
                L[indices_to_update] = torch.clip(L_update, min=1.0, max=1e6)

        # --- Inner Loop 1 (Find z via backtracking) ---
        energy_new = torch.zeros_like(energy)  # Initialize
        for ii in range(100):
            z[idx] = x_bar[idx] - grad[idx] / L[idx]
            dx = z[idx] - x_bar[idx]
            # Ensure energy and c[idx] are broadcastable for torch.max
            bound = torch.max(
                energy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # Broadcast energy
                c[idx, None, None, None],  # Ensure c[idx] is shaped correctly
            ) - delta * (dx * dx).sum((1, 2, 3), keepdim=True)

            try:
                current_energy_new = f(z[idx], y[idx])  # Evaluate f for current z
                energy_new = current_energy_new  # Assign if successful
            except Exception as e:
                print(f"Error in f(z) inner loop 1 iter {i}: {e}")
                # Let's break inner loop and proceed with caution (might need better handling)
                break

            # Compare energy_new (scalar per item) with bound (scalar per item)
            # Need to handle shapes carefully if f returns differently
            if torch.all(energy_new <= bound.view(-1)):
                break  # Condition met

            # Increase L where condition not met
            not_converged_mask = energy_new > bound.view(-1)
            L[idx[not_converged_mask]] = L[idx[not_converged_mask]] / rho
            if torch.isinf(L[idx]).any():  # Prevent L runaway
                print(
                    f"Warning: L became Inf in inner loop 1 at iter {i}. Resetting L might be needed."
                )
                # Option: Reset L, or break, or return previous state
                break  # Safer to break inner loop
        else:  # No break occurred in inner loop
            if verbose:
                print(f"Warning: Inner loop 1 max iter ({ii+1}) reached at iter {i}")

        # --- Check for second update path (non-monotone step) ---
        # dx needs to be based on the *final* z found above
        dx_z = z[idx] - x_bar[idx]
        idx2 = (
            (energy_new[:] >= (c[idx] - delta * (dx_z * dx_z).sum((1, 2, 3))))
            .nonzero()
            .view(-1)
        )

        # Initialize variables used in both branches
        final_x_update = z[idx].clone()  # Default update is z

        if idx2.nelement() > 0:
            idx_idx2 = idx[idx2]  # Original batch indices for this path
            try:
                gradx = nabla(x[idx_idx2], y[idx_idx2])
            except Exception as e:
                print(f"Error in nabla(x) inner loop 2 iter {i}: {e}")
                # If gradient fails, maybe skip this corrective step?
                gradx = None  # Signal failure

            if gradx is not None:
                # --- Lipschitz Update for this subset ---
                if i > 0:
                    # Ensure indices align: grad_old[idx_idx2], x[idx_idx2], x_bar_old[idx_idx2]
                    dx_grad_2 = gradx - grad_old[idx_idx2]
                    dx_iter_2 = (
                        x[idx_idx2] - x_bar_old[idx_idx2]
                    )  # Check if this is the intended difference
                    s2 = (dx_grad_2 * dx_grad_2).sum((1, 2, 3), keepdim=True)
                    denom2 = (dx_grad_2 * dx_iter_2).sum((1, 2, 3), keepdim=True)
                    safe_denom_mask_4d_2 = torch.abs(denom2) > 1e-8
                    safe_denom_mask_1d_2 = safe_denom_mask_4d_2.squeeze()

                    if safe_denom_mask_1d_2.any():
                        indices_to_update_2 = idx_idx2[safe_denom_mask_1d_2]
                        s2_safe = s2[safe_denom_mask_1d_2]
                        denom2_safe = denom2[safe_denom_mask_1d_2]
                        L_update_2 = s2_safe / denom2_safe
                        L[indices_to_update_2] = torch.clip(
                            L_update_2, min=1.0, max=1e6
                        )

                # --- Inner Loop 2 (Find v via backtracking) ---
                # L_old = L.clone() # Not needed if we update L in place
                v = torch.zeros_like(x[idx_idx2])  # Initialize v for this subset
                energy_new2 = torch.zeros_like(
                    energy_new[idx2]
                )  # Initialize energy for v

                for ii in range(100):
                    v = x[idx_idx2] - gradx / L[idx_idx2]
                    dx_v = v - x[idx_idx2]
                    # Ensure c[idx_idx2] is shaped correctly
                    bound2 = c[idx_idx2, None, None, None] - delta * (dx_v * dx_v).sum(
                        (1, 2, 3), keepdim=True
                    )

                    try:
                        current_energy_new2 = f(v, y[idx_idx2])
                        energy_new2 = current_energy_new2  # Assign if successful
                    except Exception as e:
                        print(f"Error in f(v) inner loop 2 iter {i}: {e}")
                        break  # Break inner loop

                    # Compare energy_new2 with bound2
                    if torch.all(energy_new2 <= bound2.view(-1) * (1 + 1e-4)):
                        break  # Condition met

                    # Increase L where condition not met
                    not_converged_mask2 = energy_new2 > bound2.view(-1) * (1 + 1e-4)
                    L[idx_idx2[not_converged_mask2]] = (
                        L[idx_idx2[not_converged_mask2]] / rho
                    )
                    if torch.isinf(L[idx_idx2]).any():  # Prevent L runaway
                        print(f"Warning: L became Inf in inner loop 2 at iter {i}.")
                        break  # Safer to break inner loop
                else:  # No break occurred in inner loop 2
                    if verbose:
                        print(
                            f"Warning: Inner loop 2 max iter ({ii+1}) reached at iter {i}"
                        )

                # --- Decide whether to use v ---
                # Compare energy from v (energy_new2) with energy from z (energy_new[idx2])
                idx3 = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
                if idx3.nelement() > 0:
                    tmp = idx_idx2[
                        idx3
                    ]  # Original batch indices where v is better or equal
                    # Update the final result for these specific indices
                    final_x_update[idx2[idx3]] = v[
                        idx3
                    ]  # Update the relevant slice of final_x_update

        # --- Update x using the result (either z or potentially v) ---
        x[idx] = final_x_update

        # --- Calculate Residual ---
        res_val = torch.zeros_like(res[idx])  # Initialize residual for active elements
        if i >= 1:  # Calculate residual from the first iteration onwards
            try:
                # Use current_x_old which stores x at the start of this iteration
                dx_res = x[idx] - current_x_old[idx]
                norm_dx_res = torch.norm(dx_res, p=2, dim=(1, 2, 3))
                norm_x = torch.norm(x[idx], p=2, dim=(1, 2, 3))
                # Avoid division by zero if norm_x is zero
                safe_norm_x = torch.where(
                    norm_x == 0, torch.tensor(1.0, device=x.device), norm_x
                )
                res_val = norm_dx_res / safe_norm_x
                res[idx] = res_val  # Update residual tracker
            except Exception as e:
                print(f"Error calculating residual at iter {i}: {e}")
                # Treat as non-converged or handle error
                res[idx] = tol + 1  # Ensure it doesn't falsely converge

        # --- Update Active Indices & Check Convergence ---
        condition = res[idx] > tol  # Check condition only on active elements
        new_idx = idx[
            condition.nonzero().view(-1)
        ]  # Get subset of idx that remains active

        # ***** CORE FIX: Check if the *new* set of active indices is empty *****
        if new_idx.nelement() == 0:
            if verbose:
                print(f"Converged in iter {i+1} (all active elements <= tol {tol:.2e})")
            iterations = i + 1
            break  # Exit main loop successfully

        # --- If not converged, update idx and continue ---
        idx = new_idx  # Update active indices for the next iteration

        t_old = t
        t = (np.sqrt(4.0 * t_old**2 + 1.0) + 1.0) / 2.0
        q_old = q
        q = eta * q + 1.0

        # --- Update c (only for active elements, idx is guaranteed non-empty here) ---
        try:
            c[idx] = (eta * q_old * c[idx] + f(x[idx], y[idx])) / q
        except Exception as e:
            print(f"Error updating 'c' at iter {i} for idx {idx}: {e}")
            pass  # Skip updating c for this iteration if f fails

        # --- Prepare for Next Iteration ---
        x_bar_old = x_bar.clone()  # Store for L update next iteration
        grad_old = grad.clone()  # Store for L update next iteration
        x_old = current_x_old  # Pass the x from the start of this iter to the next

    # --- End of Loop ---

    if (
        i == max_iter - 1 and idx.nelement() > 0
    ):  # Check if max_iter was reached with active elements
        if verbose:
            # Report max residual among currently active elements
            max_res_val = (
                torch.max(res[idx]) if idx.nelement() > 0 else torch.tensor(0.0)
            )
            print(
                f"Max iter ({max_iter}) reached, max residual among active: {max_res_val:.6f}"
            )
        iterations = max_iter  # Ensure iterations is max_iter

    if return_L:
        return x, L
    else:
        return x


def reconstruct_nmAPG_stable(
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
    return_L=False,
):
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y).to(y.device, y.dtype)

    if isinstance(lamda, (int, float)):
        effective_lamda = torch.tensor(lamda, device=y.device, dtype=y.dtype)
    elif isinstance(lamda, torch.Tensor):
        effective_lamda = lamda.to(y.device, y.dtype)
    else:
        raise TypeError("lamda must be a number or a torch.Tensor")

    # Define energy function matching the signature expected by nmAPG: f(val, y_in)
    def energy(val, y_in):
        # Ensure val is detached if needed *before* passing to potentially non-detached components
        val_maybe_detached = val.detach() if detach_grads else val

        # Calculate data fidelity term
        df_term = data_fidelity(val_maybe_detached, y_in, physics)

        # Calculate regularization term
        # Ensure regularizer.g handles potential batch dimension correctly
        reg_term = regularizer.g(val_maybe_detached)

        # Combine terms, ensure consistent shapes for addition
        # Assuming df_term and reg_term output shape (B,) or scalar that broadcasts
        fun = df_term + effective_lamda * reg_term

        # Detach final result if needed (redundant if inputs were detached unless components re-attach)
        if detach_grads:
            fun = fun.detach()

        # Ensure output shape is (B,) for compatibility with nmAPG internal logic (like c[idx])
        return fun.reshape(-1)

    # Define gradient function matching the signature expected by nmAPG: nabla(val, y_in)
    def energy_grad(val, y_in):
        # Enable gradient calculation for val if detached previously or if requires_grad is False
        val_for_grad = (
            val.detach().requires_grad_(True) if not val.requires_grad else val
        )

        # Calculate gradient of data fidelity
        grad_df = data_fidelity.grad(val_for_grad, y_in, physics)

        # Calculate gradient of regularizer
        grad_reg = regularizer.grad(val_for_grad)

        # Combine gradients
        grad = grad_df + effective_lamda * grad_reg

        if detach_grads:
            grad = grad.detach()
        return grad

    # Call nmAPG
    # Assuming L_init = 1/step_size is intended, though step_size is not directly used by nmAPG itself
    result = nmAPG_stable(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        L_init=1.0 / step_size if step_size > 0 else 1.0,  # Handle step_size=0
        tol=tol,
        verbose=verbose,
        return_L=return_L,
    )

    # Unpack result based on return_L
    if return_L:
        rec, L_final = result

        iterations = max_iter  # Placeholder
        return rec, L_final  # Or modify nmAPG to return (rec, iterations, L_final)
    else:
        rec = result
        if torch.isnan(rec).any():
            rec = torch.full_like(rec, float("inf"))
        return rec
