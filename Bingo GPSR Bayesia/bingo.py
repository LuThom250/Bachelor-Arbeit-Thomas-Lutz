# Alle Importe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import time
import random
from tqdm import tqdm
import sympy as sp
from tabulate import tabulate
from colorama import Fore, Style
import casadi as ca
from scipy import optimize

# NEU: Import für Bingo (vereinfacht)
from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def print_highlighted(text, color=Fore.CYAN):
    print(color + text + Style.RESET_ALL)

# -----------------------------------------------------------------------------
# 1. PINN SETUP FOR BEAM (Dein CasADi-Code, unverändert)
# -----------------------------------------------------------------------------
prob_para = {"E": 150e9, "I": 40e-6, "S": 60e3, "L": 2.5}
E, I, S, L = prob_para['E'], prob_para['I'], prob_para['S'], prob_para['L']
config = {"layers": [1, 4, 8, 12, 8, 4, 1], "N_f": 25, "lr": 1e-3, "n_epochs": 50000, "min_error": 1e-6}

def ada_max_update(params, grads, m, v, t,
                   beta1=0.9, beta2=0.999, alpha=0.001, eps=1e-8):
    for i in range(len(params)):
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i]    = ca.fmax(beta2 * v[i], ca.fabs(grads[i]))
        alpha_t = alpha / (1 - beta1**t)
        params[i] = params[i] - alpha_t * m[i] / (v[i] + eps)

def run_pinn_simulation():
    print_highlighted("Phase 1: Starte PINN-Simulation...", Fore.CYAN)
    t0 = time.time()
    x = ca.SX.sym('x')

    W, I, E, L_param = prob_para['S'], prob_para['I'], prob_para['E'], prob_para['L']

    sizes = config['layers']
    sym_vars = []
    for i in range(len(sizes) - 1):
        sym_vars.append(ca.SX.sym(f'w{i+1}', sizes[i+1], sizes[i]))
        sym_vars.append(ca.SX.sym(f'b{i+1}', 1, sizes[i+1]))

    def layer(inp, w, b):
        z = ca.mtimes(inp, w.T) + b
        return ca.tanh(z)

    n_out = x
    for i in range(len(sizes) - 1):
        n_out = layer(n_out, sym_vars[2*i], sym_vars[2*i+1])

    dNN = n_out

    d1NN = ca.jacobian(dNN, x); d2NN = ca.jacobian(d1NN, x)
    d3NN = ca.jacobian(d2NN, x); d4NN = ca.jacobian(d3NN, x)

    n_xi = config['N_f']
    collocation_points = np.linspace(0, L, n_xi)
    col_err = sum([(ca.substitute(d4NN, x, xi) - W/(E*I))**2 for xi in collocation_points])
    bc_err = (ca.substitute(dNN, x, 0)**2 + ca.substitute(dNN, x, L)**2 +
              ca.substitute(d2NN, x, 0)**2 + ca.substitute(d2NN, x, L)**2)
    ZF_sum = col_err/n_xi + bc_err

    f_ZF = ca.Function('f_ZF', sym_vars, [ZF_sum])
    f_grads = ca.Function('f_grads', sym_vars, [ca.gradient(ZF_sum, v) for v in sym_vars])

    vals = [ca.DM.rand(*v.size()) * 0.1 for v in sym_vars]
    m = [ca.DM.zeros(*v.size()) for v in sym_vars]
    v = [ca.DM.zeros(*v.size()) for v in sym_vars]

    max_epochs = config['n_epochs']; min_error = config['min_error']; t = 0
    
    pbar = tqdm(range(max_epochs), desc="PINN Training")
    for epoch in pbar:
        t += 1
        zf_val = float(f_ZF(*vals)[0])
        pbar.set_description(f"PINN Epoche {t}: Verlust = {zf_val:.3e}")
        if zf_val <= min_error: break
        grad_vals = f_grads(*vals)
        ada_max_update(vals, grad_vals, m, v, t, alpha=config['lr'])

    print(f"PINN Training beendet nach {t} Epochen. Laufzeit: {time.time() - t0:.2f}s")
    
    f_u = ca.Function('f_u', [x] + sym_vars, [dNN])
    xs = np.linspace(0, L, 50)
    ys = np.array([float(f_u(xi, *vals)[0]) for xi in xs])
    
    return xs.reshape(-1, 1), ys

# ---------------------------------------
# Bingo-Wrapper-Funktion
# ---------------------------------------
# ---------------------------------------
# NEU: Bingo-Wrapper-Funktion (KORRIGIERT)
# ---------------------------------------
# ---------------------------------------
# NEU: Bingo-Wrapper-Funktion (KORRIGIERT und VEREINFACHT)
# ---------------------------------------
def run_bingo_on_dataset(X, y, population_size=100, stack_size=32, generations=500):
    """Führt Bingo auf dem Dataset (X, y) aus."""

    # Konfiguration für Bingo
    regressor = SymbolicRegressor(
        population_size=population_size,
        stack_size=stack_size,
        use_simplification=True,
        crossover_prob=0.4,
        mutation_prob=0.4,
        generations=generations  # Generationen als Parameter beim Initialisieren
    )
    
    print(f"Starte Bingo mit {generations} Generationen...")
    # fit() ohne generations Parameter
    regressor.fit(X, y)
    
    # Bestes Ergebnis extrahieren
    best_individual = regressor.get_best_individual()
    equation = str(best_individual)
    mse = best_individual.fitness
    
    return regressor, equation, mse

# ---------------------------------------
# Main script
# ---------------------------------------
def main():
    # 1) Trainiere das Balken-PINN
    try:
        data = np.loadtxt("PINNBeamForward_py.txt")
        X_pinn, y_pinn = data[:,0].reshape(-1,1), data[:,1]
        print_highlighted("PINN-Daten aus Datei geladen.", Fore.YELLOW)
    except IOError:
        X_pinn, y_pinn = run_pinn_simulation()

    # 2) Führe Bingo auf den PINN-Daten aus
    print_highlighted("\n=== Starte Benchmark: Bingo auf Balken-Daten ===", Fore.CYAN)
    
    model, equation, mse = run_bingo_on_dataset(X_pinn, y_pinn)
    
    print_highlighted("\n--- ERGEBNIS FÜR BINGO ---", Fore.GREEN)
    print(f"→ Bester gefundener Ausdruck: {equation}")
    print(f"→ Trainings-MSE: {mse:.6g}")

    # 3) Plot des Ergebnisses
    xs_plot = np.linspace(0.0, L, 200)
    y_true_plot = np.interp(xs_plot, X_pinn.flatten(), y_pinn)
    
    try:
        y_pred_plot = model.predict(xs_plot.reshape(-1,1))
        # Stelle sicher, dass y_pred_plot 1D ist
        if y_pred_plot.ndim > 1:
            y_pred_plot = y_pred_plot.flatten()
    except Exception:
        try:
            x_sym = sp.Symbol('X_0')
            sym_expr = sp.sympify(equation.replace(')(', ')*(')) # Sicherheitsfix für Sympy
            lambda_expr = sp.lambdify(x_sym, sym_expr, 'numpy')
            y_pred_plot = lambda_expr(xs_plot)
            # Stelle sicher, dass y_pred_plot 1D ist
            if hasattr(y_pred_plot, 'ndim') and y_pred_plot.ndim > 1:
                y_pred_plot = y_pred_plot.flatten()
        except Exception:
            y_pred_plot = np.full_like(xs_plot, np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(xs_plot, y_true_plot, '-', label="PINN u(x) (Ziel)", lw=2.5)
    
    # Robustes Plotting mit Mask
    if y_pred_plot is not None:
        # Stelle sicher, dass beide Arrays 1D sind und gleiche Länge haben
        y_pred_plot = np.asarray(y_pred_plot).flatten()
        xs_plot_flat = np.asarray(xs_plot).flatten()
        
        # Nur plotten wenn Arrays gleiche Länge haben
        if len(y_pred_plot) == len(xs_plot_flat):
            mask = np.isfinite(y_pred_plot)
            if np.any(mask):
                plt.plot(xs_plot_flat[mask], y_pred_plot[mask], '--', label=f"Bingo: {equation[:50]}...", lw=2)
    
    plt.xlabel("x [m]"); plt.ylabel("u(x) [m]"); plt.title(f"Bingo auf Balken-Daten (MSE={mse:.4g})")
    plt.legend(); plt.grid(alpha=0.3); plt.show()

if __name__ == "__main__":
    main()