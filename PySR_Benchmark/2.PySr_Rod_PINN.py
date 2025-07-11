from pysr import install as pysr_install, PySRRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
from scipy.interpolate import interp1d
from scipy import optimize
from tqdm import tqdm
import re
import sympy as sp
# einmalig, wenn noch nicht geschehen
pysr_install()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------
# PINN for rod ODE: E A u''(x) + q0 * x = 0
# BC: u(0)=0, u'(L)=0
# Analytical: u(x) = -q0/(6EA)*x^3 + q0 L^2/(2EA) * x
# ---------------------------------------
# Problem parameters
E = 100.0
A = 1.0
q0 = 10.0
L = 5.0

# PINN hyperparameters
N_f = 100
layers = [1, 4, 8, 12, 8, 4, 1]
learning_rate = 1e-3
n_epochs = 10000
ACTIVATION_FN = nn.Tanh()

# Collocation points
x_f = torch.linspace(0, L, N_f, requires_grad=True).view(-1,1).to(device)

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = ACTIVATION_FN

    def forward(self, x):
        out = x
        for i in range(len(self.layers)-1):
            out = self.layers[i](out)
            out = self.activation(out)
        out = self.layers[-1](out)
        return out

def pde_residual(model, x):
    # model: u(x)
    u = model(x)
    # first derivative
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # second derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    # Residual: A*E*u_xx + q0 * x = 0
    return A * E * u_xx + q0 * x

def loss_function(model, x_f):
    # PDE residual loss
    f = pde_residual(model, x_f)
    loss_f = torch.mean(f**2)
    # BC at x=0: u(0)=0
    x0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    u0 = model(x0)
    loss_bc1 = u0**2
    # BC at x=L: u'(L)=0
    xL = torch.tensor([[L]], requires_grad=True).to(device)
    uL = model(xL)
    uL_x = torch.autograd.grad(uL, xL, grad_outputs=torch.ones_like(uL), create_graph=True)[0]
    loss_bc2 = uL_x**2
    return loss_f + loss_bc1 + loss_bc2

def train_pinn():
    model = PINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(1, n_epochs+1):
        optimizer.zero_grad()
        loss = loss_function(model, x_f)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6e}")
    return model, loss_history

def generate_pinn_dataset(pinn_model, n_points=50):
    """
    Generate dataset (X_pinn, y_pinn) from PINN:
    X_pinn: shape (n_points,1) numpy; y_pinn: (n_points,) numpy
    """
    xs = np.linspace(0.0, L, n_points)
    pinn_model.eval()
    with torch.no_grad():
        xt = torch.tensor(xs.reshape(-1,1), dtype=torch.float32).to(device)
        y_tensor = pinn_model(xt).detach().cpu().numpy().flatten()
    X_pinn = xs.reshape(-1,1)
    y_pinn = y_tensor
    return X_pinn, y_pinn

# ---------------------------------------
# PySR helper functions
# ---------------------------------------
def map_ops_to_pysr(ops_list):
    """
    Map operator list like ['+', '-', '*', '^', 'sin', ...]
    to PySR's binary_operators / unary_operators.
    '^' -> 'pow'; unary: 'sin','cos','exp'.
    """
    binary_map = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "pow"}
    unary_map  = {"sin": "sin", "cos": "cos", "exp": "exp"}
    bin_ops = []
    un_ops = []
    for op in ops_list:
        if op in binary_map:
            bin_ops.append(binary_map[op])
        if op in unary_map:
            un_ops.append(unary_map[op])
    if not bin_ops:
        bin_ops = None
    if not un_ops:
        un_ops = None
    return bin_ops, un_ops

def run_pysr_on_dataset(X, y, ops,
                        variable_names=["x0"],
                        niterations=40,
                        population_size=50,
                        random_state=None):
    """
    Fit PySRRegressor on (X, y) with operator set `ops`.
    Returns: model, best expression string, simplified expression string, training MSE.
    """
    bin_ops, un_ops = map_ops_to_pysr(ops)
    model = PySRRegressor(
        niterations=niterations,
        population_size=population_size,
        binary_operators=bin_ops,
        unary_operators=un_ops,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        verbosity=1,
        random_state=random_state,
    )
    model.fit(X, y, variable_names=variable_names)
    best = model.get_best()
    expr_pysr = best.get("sympy_format", None)
    simplified_expr = None
    if expr_pysr is not None:
        try:
            # Parse and simplify using sympy
            sym_expr = sp.sympify(expr_pysr)
            simp = sp.simplify(sym_expr)
            simplified_expr = str(simp)
        except Exception as e:
            print(f"Sympy simplification failed: {e}")
            simplified_expr = expr_pysr
    try:
        y_pred_train = model.predict(X)
        mse = float(np.mean((y - y_pred_train)**2))
    except Exception:
        mse = float("nan")
    return model, expr_pysr, simplified_expr, mse

# ---------------------------------------
# Main script
# ---------------------------------------
def main():
    # 1) Train the PINN
    print("Training PINN for rod ODE...")
    pinn_model, loss_hist = train_pinn()

    # 2) Generate dataset from PINN
    X_pinn, y_pinn = generate_pinn_dataset(pinn_model, n_points=50)
    print("Generated PINN dataset: X_pinn.shape =", X_pinn.shape, "y_pinn.shape =", y_pinn.shape)

    # 3) Run PySR on PINN data with different operator sets
    operator_sets = [
        (['+', '-', '*', '^'], "Poly only"),
        (['+', '-', '*', '^', 'sin', 'cos'], "Poly + trig"),
        (['+', '-', '*', '^', 'sin', 'cos', 'exp'], "Poly + trig + exp"),
    ]

    results = {}
    for ops, label in operator_sets:
        print(f"\n=== PySR on PINN data with operators: {label} ===")
        model, expr_pysr, simplified_expr, mse_train = run_pysr_on_dataset(
            X_pinn, y_pinn, ops,
            variable_names=["x0"],
            niterations=40,
            population_size=50,
            random_state=0
        )
        print(f"Found expression: {expr_pysr}")
        if simplified_expr is not None:
            print(f"Simplified expression: {simplified_expr}")
        print(f"Training MSE: {mse_train:.6g}")
        results[label] = (expr_pysr, simplified_expr, mse_train, model)

        # Plot comparison: PINN vs PySR approx
        xs_plot = np.linspace(0.0, L, 200)
        # PINN predictions
        pinn_model.eval()
        with torch.no_grad():
            xt_plot = torch.tensor(xs_plot.reshape(-1,1), dtype=torch.float32).to(device)
            y_true_plot = pinn_model(xt_plot).detach().cpu().numpy().flatten()
        # PySR predictions
        try:
            y_pred_plot = model.predict(xs_plot.reshape(-1,1))
            y_pred_plot = np.array(y_pred_plot, dtype=float)
        except Exception:
            # fallback: eval expr_pysr
            y_pred_list = []
            if expr_pysr is not None:
                for xv in xs_plot:
                    try:
                        val = eval(expr_pysr,
                                   {"__builtins__": {}},
                                   {"x0": xv, "sin": math.sin, "cos": math.cos, "exp": math.exp})
                    except Exception:
                        val = float('nan')
                    y_pred_list.append(val)
                y_pred_plot = np.array(y_pred_list, dtype=float)
            else:
                y_pred_plot = np.full_like(xs_plot, np.nan)

        plt.figure(figsize=(6,4))
        plt.plot(xs_plot, y_true_plot, '-', label="PINN u(x)", lw=2)
        mask = np.isfinite(y_pred_plot)
        if mask.any():
            plt.plot(xs_plot[mask], y_pred_plot[mask], '--', label=f"PySR ≈ {simplified_expr or expr_pysr}", lw=2)
        plt.xlabel("x"); plt.ylabel("u(x)")
        plt.title(f"PySR on PINN data ({label}), MSE={mse_train:.4g}")
        plt.legend(); plt.grid(alpha=0.3)
        plt.show()

    # 4) Summary
    print("\n=== Summary of PySR results on PINN data ===")
    for label, (expr_pysr, simplified_expr, mse_train, _) in results.items():
        expr_str = str(simplified_expr) if simplified_expr is not None else str(expr_pysr)
        print(f"{label:<15} | Expr: {expr_str:<40} | MSE: {mse_train:.6g}")

if __name__ == "__main__":
    main()
