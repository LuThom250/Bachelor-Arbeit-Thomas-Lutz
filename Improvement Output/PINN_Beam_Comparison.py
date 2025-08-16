# Alle Importe
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

def print_highlighted(message, color=Fore.GREEN):
    """Gibt eine Nachricht farbig hervorgehoben aus."""
    print(color + message + Style.RESET_ALL)

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
    "min_error":     1e-7,     # Wenn kleiner als dieser Error Training beenden
}
N_f            = config['N_f']                      # Kollokations­punkte (Kompromiss)
layers         = config['layers']        # Netz­architektur (ausreichend komplex)
learning_rate  = config['lr']                   # Lernrate (moderat erhöht)
n_epochs       = config['n_epochs']                    # Anzahl der Trainings­epochen (Kompromiss)

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


# =======================================================
# HINZUFÜGEN: Plot-Funktionen zur Visualisierung
# =======================================================
def plot_solutions(x_values, pinn_solution, analytical_solution):
    """Plot comparison between PINN solution and analytical solution."""
    plt.figure(figsize=(12, 7))
    plt.plot(x_values, analytical_solution, 'k-', linewidth=3, label='Analytical solution')
    plt.plot(x_values, pinn_solution, 'r--', marker='o', markersize=5, label='PINN solution')
    plt.title('Beam Deflection: PINN vs Analytical', fontsize=24)
    plt.xlabel('x [m]', fontsize=22)
    plt.ylabel('Deflection u(x) [m]', fontsize=22)
    plt.legend(fontsize=20, loc='best')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_training_loss(error_log):
    """Plot training loss over epochs (log scale)."""
    epochs, errors = zip(*error_log)
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, errors, color='red', linewidth=2, label='Training Loss')
    plt.yscale('log')
    plt.title('Training Loss History', fontsize=24)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss (log scale)', fontsize=22)
    plt.legend(fontsize=20, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_absolute_error(x_values, pinn_solution, analytical_solution):
    """Plot absolute error between PINN and analytical solutions."""
    error = np.abs(np.array(pinn_solution) - np.array(analytical_solution))
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, error, 'b-', linewidth=2, label='Absolute error')
    plt.title('Absolute Error: PINN vs Analytical', fontsize=24)
    plt.xlabel('x [m]', fontsize=22)
    plt.ylabel('Absolute error [m]', fontsize=22)
    plt.legend(fontsize=20, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =======================================================
# HINZUFÜGEN: Dieser Block startet die Simulation
# =======================================================
# =======================================================
# HINZUFÜGEN: Dieser Block startet die Simulation
# =======================================================
if __name__ == "__main__":
    # Run the main simulation function
    run_pinn_simulation() 
    
    # Load the saved data for visualization
    error_log_data = np.loadtxt("ErrorFileBeamForward_py.txt")
    pinn_solution_data = np.loadtxt("PINNBeamForward_py.txt")
    analytical_solution_data = np.loadtxt("LSGBeamForward_py.txt")

    # Plot the results
    plot_training_loss(error_log_data)
    plot_solutions(
        pinn_solution_data[:, 0], # x-values
        pinn_solution_data[:, 1], # PINN u(x)
        analytical_solution_data[:, 1] # Analytical u(x)
    )
    
    # HIER DEN NEUEN AUFRUF EINFÜGEN
    plot_absolute_error(
        pinn_solution_data[:, 0], # x-values
        pinn_solution_data[:, 1], # PINN u(x)
        analytical_solution_data[:, 1] # Analytical u(x)
    )