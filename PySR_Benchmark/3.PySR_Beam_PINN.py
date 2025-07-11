# Alle Importe
# PySR Import
from pysr import PySRRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import torch.nn.functional as F
import torch.optim as optim
import re
from collections import deque
import random
from tqdm import tqdm
import sys
import sympy as sp
from tabulate import tabulate
from colorama import Fore, Style
from scipy import optimize



sys.setrecursionlimit(10000)

import casadi as ca
import time

# ==============================
# Problemparameter DGL (korrigiert nach Bachelorarbeit)
# ==============================
prob_para = {
    "E" : 150e9     , # Elastizitätsmodul [Pa] - aus Bachelorarbeit
    "I": 40e-6      ,  # Flächenträgheitsmoment [m^4] - aus Bachelorarbeit
    "S": 60e3       ,  # Belastungskonstante [N/m] - aus Bachelorarbeit
    "L": 2.5        ,  # Länge des Balkens [m] - aus Bachelorarbeit
}

E   = prob_para['E']
I   = prob_para['I']
S   = prob_para['S']
L   = prob_para['L']

# ==============================
# Wichtige Hyperparameter PINN (ausbalanciert für Geschwindigkeit + Qualität)
# ==============================
config = {
    "layers": [1, 4, 8, 12, 8, 4, 1],
    # Trainingsdaten
    "N_f":  25       ,      # Anzahl der Innen-Stützpunkte
    # Training
    "lr":            1e-3,  # Lernrate
    "n_epochs":      50000,     # Anzahl der Trainings-Epochen
    "min_error":     1e-6,     # Wenn kleiner als dieser Error Training beenden
}
N_f            = config['N_f']                      # Kollokations­punkte (Kompromiss)
layers         = config['layers']        # Netz­architektur (ausreichend komplex)
learning_rate  = config['lr']                   # Lernrate (moderat erhöht)
n_epochs       = config['n_epochs']                    # Anzahl der Trainings­epochen (Kompromiss)

def print_highlighted(text, color=Fore.GREEN):
    """Hilfsfunktion für farbige Ausgabe"""
    print(color + text + Style.RESET_ALL)

def ada_max_update(params, grads, m, v, t,
                   beta1=0.9, beta2=0.999, alpha=0.001, eps=1e-8):
    """
    AdaMax-Update für Listen von casadi.DM-Parametern und Gradienten.
    m, v sind Listen gleicher Länge wie params bzw. grads.
    """
    for i in range(len(params)):
        # Momentumschätzung
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        # Infinity-Norm-Schätzung
        grad_abs = ca.fabs(grads[i])
        v[i]    = ca.fmax(beta2 * v[i], grad_abs)
        # bias-korrigierte Lernrate
        alpha_t = alpha / (1 - beta1**t)
        # Parameter-Update
        params[i] = params[i] - alpha_t * m[i] / (v[i] + eps)

def run_pinn_simulation():
    """
    Führt die gesamte PINN-Simulation aus, um die Biegelinie zu approximieren.
    Gibt die trainierten Parameter und Funktionen zurück für die Datengenerierung.
    """
    print_highlighted("Phase 1: Starte PINN-Simulation...", Fore.CYAN)
    # ---- 1) Setup ----
    # Startzeit erfassen
    t0 = time.time()
    # Symbolische Variable
    x = ca.SX.sym('x')

    # Parameter
    W = prob_para['S']
    I = prob_para['I']
    E = prob_para['E']
    L = prob_para['L']

    # Analytische Lösung (4. Ableitung)
    DGL    = W/(24*E*I) * (-x**4 + 2*L*x**3 - L**3 * x)
    D1     = ca.jacobian(DGL, x)
    D2     = ca.jacobian(D1,  x)
    D3     = ca.jacobian(D2,  x)
    D4     = ca.jacobian(D3,  x)

    # Kollokationspunkte
    n_xi = config['N_f']
    collocation_points = np.linspace(0, L, n_xi)

    # ---- 2) PINN-Netzwerk definieren ----
    # Layer-Größen
    sizes = config['layers']
    # Gewichte & Bias als SX-Symbole
    w1    = ca.SX.sym('w1',  sizes[0], sizes[1])
    b1    = ca.SX.sym('b1',  1, sizes[1])
    w2    = ca.SX.sym('w2',  sizes[1], sizes[2])
    b2    = ca.SX.sym('b2',  1, sizes[2])
    w3    = ca.SX.sym('w3',  sizes[2], sizes[3])
    b3    = ca.SX.sym('b3',  1, sizes[3])
    w4    = ca.SX.sym('w4',  sizes[3], sizes[4])
    b4    = ca.SX.sym('b4',  1, sizes[4])
    w5    = ca.SX.sym('w5',  sizes[4], sizes[5])
    b5    = ca.SX.sym('b5',  1, sizes[5])
    w_end = ca.SX.sym('w_end', sizes[5], sizes[6])
    b_end = ca.SX.sym('b_end', 1, sizes[6])

    # Forward-pass
    def layer(inp, w, b):
        z = inp @ w + b
        return 0.5 * (1 + ca.tanh(z / 2))

    n1s = layer(x, w1, b1)
    n2s = layer(n1s, w2, b2)
    n3s = layer(n2s, w3, b3)
    n4s = layer(n3s, w4, b4)
    n5s = layer(n4s, w5, b5)
    n_end = n5s @ w_end + b_end
    dNN   = n_end

    # Ableitungen des NN
    d1NN = ca.jacobian(dNN, x)
    d2NN = ca.jacobian(d1NN, x)
    d3NN = ca.jacobian(d2NN, x)
    d4NN = ca.jacobian(d3NN, x)

    # ---- 3) Zielfunktion aufbauen ----
    # Kollokationsfehler (D4 u(x) + W/(E*I) = 0)
    col_err = 0
    for xi in collocation_points:
        col_err += (ca.substitute(d4NN, x, xi) + W/(E*I))**2

    # Randbedingungen u(0)=u(L)=u''(0)=u''(L)=0
    bc_err = (
        ca.substitute(dNN,  x, 0)**2 +
        ca.substitute(dNN,  x, L)**2 +
        ca.substitute(d2NN, x, 0)**2 +
        ca.substitute(d2NN, x, L)**2
    )

    ZF_sum = col_err/n_xi + bc_err

    # ---- 4) CasADi-Funktionen für Eval und Grad ----
    sym_vars = [w1, w2, w3, w4, w5, w_end,
                b1, b2, b3, b4, b5, b_end]

    # Zielfunktion
    f_ZF = ca.Function('f_ZF', sym_vars, [ZF_sum])

    # Gradienten jeder Gewichtsmatrix / jedes Bias
    grads = []
    for v in sym_vars:
        g = ca.jacobian(ZF_sum, v)
        g = ca.reshape(g, v.size1(), v.size2())
        grads.append(g)
    f_grads = ca.Function('f_grads', sym_vars, grads)

    # ---- 5) Parameter initialisieren ----
    # zufällige Startwerte
    vals = [ca.DM.rand(*v.size()) for v in sym_vars]
    # Moment- und Varianzspeicher für AdaMax
    m = [ca.DM.zeros(*v.size()) for v in sym_vars]
    v = [ca.DM.zeros(*v.size()) for v in sym_vars]

    # Hyperparam
    max_epochs = config['n_epochs']
    min_error  = config['min_error']
    t = 0

    # Logs
    error_log = []

    # ---- 6) Training ----
    while True:
        t += 1
        zf_val = float(f_ZF(*vals)[0])
        error_log.append((t, zf_val))

        if t % 500 == 0:
            print(f"Epoch {t:6d}, ZF = {zf_val:.3e}")

        if zf_val <= min_error or t >= max_epochs:
            print(f"Training beendet bei Epoch {t}, ZF = {zf_val:.3e}")
            break

        # Gradienten berechnen
        grad_vals = f_grads(*vals)
        # AdaMax-Update
        ada_max_update(vals, grad_vals, m, v, t)

    # ---- 7) Ergebnisse speichern ----
    np.savetxt("ErrorFileBeamForward_py.txt", np.array(error_log))

    # NN-Lösung an den Kollokationspunkten
    f_u = ca.Function('f_u', [x] + sym_vars, [dNN])
    sol_u = [float(f_u(xi, *vals)[0]) for xi in collocation_points]
    np.savetxt("PINNBeamForward_py.txt",
               np.vstack([collocation_points, sol_u]).T)

    # Analytische Lösung
    ana_u = [float(ca.substitute(DGL, x, xi)) for xi in collocation_points]
    np.savetxt("LSGBeamForward_py.txt",
               np.vstack([collocation_points, ana_u]).T)

    # Laufzeit
    print(f"Ausführungszeit PINN: {time.time() - t0:.2f}s")
    print_highlighted("PINN-Simulation abgeschlossen.", Fore.CYAN)
    
    # Return trained model components for data generation
    return f_u, vals, sym_vars, x

def generate_pinn_dataset(f_u, vals, sym_vars, x, n_points=50):
    """
    Generate dataset (X_pinn, y_pinn) from trained PINN:
    X_pinn: shape (n_points,1) numpy; y_pinn: (n_points,) numpy
    """
    xs = np.linspace(0.0, L, n_points)
    y_pinn = []
    
    for xi in xs:
        u_val = float(f_u(xi, *vals)[0])
        y_pinn.append(u_val)
    
    X_pinn = xs.reshape(-1, 1)
    y_pinn = np.array(y_pinn)
    
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
    global L, prob_para
    # 1) Train the PINN for beam bending
    print_highlighted("=== Training PINN for Beam Bending ===", Fore.MAGENTA)
    f_u, vals, sym_vars, x = run_pinn_simulation()

    # 2) Generate dataset from PINN
    print_highlighted("=== Generating Dataset from PINN ===", Fore.YELLOW)
    X_pinn, y_pinn = generate_pinn_dataset(f_u, vals, sym_vars, x, n_points=50)
    print(f"Generated PINN dataset: X_pinn.shape = {X_pinn.shape}, y_pinn.shape = {y_pinn.shape}")

    # 3) Run PySR on PINN data with different operator sets
    print_highlighted("=== Running PySR on PINN Data ===", Fore.GREEN)
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
        y_true_plot = []
        for xi in xs_plot:
            u_val = float(f_u(xi, *vals)[0])
            y_true_plot.append(u_val)
        y_true_plot = np.array(y_true_plot)
        
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

        plt.figure(figsize=(10, 6))
        plt.plot(xs_plot, y_true_plot, '-', label="PINN u(x)", lw=2)
        mask = np.isfinite(y_pred_plot)
        if mask.any():
            plt.plot(xs_plot[mask], y_pred_plot[mask], '--', 
                    label=f"PySR ≈ {simplified_expr or expr_pysr}", lw=2)
        plt.xlabel("x [m]")
        plt.ylabel("u(x) [m]")
        plt.title(f"Beam Deflection: PySR on PINN data ({label}), MSE={mse_train:.4g}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # 4) Summary
    print_highlighted("=== Summary of PySR results on PINN Beam data ===", Fore.CYAN)
    for label, (expr_pysr, simplified_expr, mse_train, _) in results.items():
        expr_str = str(simplified_expr) if simplified_expr is not None else str(expr_pysr)
        print(f"{label:<20} | Expr: {expr_str:<50} | MSE: {mse_train:.6g}")

    # 5) Compare with analytical solution
    print_highlighted("=== Comparison with Analytical Solution ===", Fore.MAGENTA)
    xs_anal = np.linspace(0, L, 200)
    W = prob_para['S']
    I = prob_para['I']
    E = prob_para['E']
    L = prob_para['L']
    
    # Analytical solution: u(x) = W/(24*E*I) * (-x^4 + 2*L*x^3 - L^3*x)
    y_analytical = W/(24*E*I) * (-xs_anal**4 + 2*L*xs_anal**3 - L**3*xs_anal)
    
    # PINN predictions for comparison
    y_pinn_comp = []
    for xi in xs_anal:
        u_val = float(f_u(xi, *vals)[0])
        y_pinn_comp.append(u_val)
    y_pinn_comp = np.array(y_pinn_comp)
    
    plt.figure(figsize=(12, 8))
    plt.plot(xs_anal, y_analytical, '-', label="Analytical Solution", lw=3, color='black')
    plt.plot(xs_anal, y_pinn_comp, '--', label="PINN Solution", lw=2, color='red')
    
    # Add best PySR result
    best_label = min(results.keys(), key=lambda k: results[k][2])  # Lowest MSE
    best_model = results[best_label][3]
    try:
        y_best_pysr = best_model.predict(xs_anal.reshape(-1,1))
        plt.plot(xs_anal, y_best_pysr, ':', label=f"Best PySR ({best_label})", lw=2, color='blue')
    except:
        pass
    
    plt.xlabel("x [m]")
    plt.ylabel("u(x) [m]")
    plt.title("Beam Deflection: Comparison of Solutions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()