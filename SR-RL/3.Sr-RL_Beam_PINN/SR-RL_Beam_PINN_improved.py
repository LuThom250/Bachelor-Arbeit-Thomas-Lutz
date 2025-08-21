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

# -------------------------------------------------
# Hyperparamter für Symbolic Policy with Reinforcement Learning
# -------------------------------------------------
MAX_LENGTH    = 15      # Maximale Token-Länge eines Ausdrucks, odt auch 12
X_RANGE       = (0, L)  # Betrachtung des Intervalls, muss mit L übereinstimmen
N_POINTS      = 50      # Anzahl x-Stützstellen pro Episode
EPISODES      = 10000   # Gesamtzahl RL-Episoden, standard 10000
BATCH_SIZE    = 64      # Minibatch-Größe für Replay-Updates
LR            = 0.001   # Lernrate des Policy-Netzes
EMBED_SIZE    = 64      # Embedding dimension
HIDDEN_SIZE   = 128     # Hidden layer size
GAMMA         = 0.99    # Discount factor
EPSILON_START = 1.0     # Initial exploration rate, Parameter der ε-greedy Exploration
EPSILON_END   = 0.2     # Final exploration rate, Parameter der ε-greedy Exploration
EPSILON_DECAY = 10000   # Exploration decay steps, Parameter der ε-greedy Exploration, standrad 5000-10000
ENTROPY_COEF  = 0.3     # (derzeit ungenutzt) Entropie-Bonus für stochastische Policies
MEMORY_SIZE   = 10000   # Kapazität des Replay Buffers
POLICY_UPDATE_FREQ = 5  # wie oft (in Episoden) das Policy-Netz trainiert wird
TARGET_UPDATE_FREQ = 50 # wie oft die Target-Netz-Gewichte synchronisiert werden

# ------- Sicherheits-/Stabilitätsparameter --------------------------
SAFE_DIV_EPS  = 1e-6        # Division-Absicherung, Kleiner Wert, der Nenner 0 in Division ersetzt
EXP_CLAMP     =  (0, 20)  # Overflow-Schwelle für exp-Aufruf, Wertebereich, auf den Argumente von exp() begrenzt werden
BASE_CLAMP    = (0, 100) # Overflow / NaN-Schutz, für Bsis Exponent, Wertebereich der Basis bei Potenz-Operation
POW_CLAMP      = (0, 6)    # Begrenzung Exponent in ^, erlaubter Exponenten-Bereich
GRAD_CLIP      =1.0         # Gradienren_Clipping, um zu große Updates zu veremiden, wenn größer oder kleiner -1, dann auf max 1 oder -1 gesetzt,Komponentenweises Gradient-Clipping (±GRAD_CLIP)
BONUS_SCALE    =0.01        # Einfluss der Verbesserungs-Prämie, Gewicht des relativen Verbesserungs-Bonus im Reward

# ------- Konstantenoptimierung --------------------------------------
CONST_DECIMALS = 3                     # Rundungsgenauigkeit (2 → 0.01-Raster), # 0 = ganze Zahl, 1 = Zehntel 0.1 , 2 = Hundertstel 0.01 ...
CONST_FACTOR   = 10 ** CONST_DECIMALS  # Hilfsfaktor zum Runden
CONST_RANGE = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.14, -3.14, 0.25, -0.25, 5.0, -5.0, 10.0, -10.0, math.pi, math.e] # kann man beliebig anpassen, Liste zulässiger Konstanten, wenn der zu Beginn der Optimierung der Agent „TOK_CONST“ wählt
MAXITER_OPT    =50                    # max. Iterationen für SciPy-minimize

# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def print_table(data, headers=["Variable", "Wert"]):
    """Gibt die Daten als übersichtliche Tabelle aus."""
    print(tabulate(data, headers=headers, tablefmt="grid"))

def print_highlighted(message, color=Fore.GREEN):
    """Gibt eine Nachricht farbig hervorgehoben aus."""
    print(color + message + Style.RESET_ALL)

def enhanced_progress_bar(iterable, desc="Verarbeitung läuft", unit=" Step"):
    """Ersetzt die Standard-tqdm-Schleife durch eine erweiterte Fortschrittsanzeige."""
    return tqdm(iterable, desc=desc, unit=unit)

# -------------------------------------------------
# SAFETY FUNCTIONS
# -------------------------------------------------
SAFE_FUNCS = {
    'sin': torch.sin,
    'cos': torch.cos,
    'exp': torch.exp,
    'abs': torch.abs,
    'pow': torch.pow,
}

# -------------------------------------------------
# TOKENIZATION
# -------------------------------------------------
TOKEN_PATTERN = r"\s*(sin|cos|exp|[()+\-*/^]|[0-9]+(?:\.[0-9]+)?|x)"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

def tokenize(expr):
    """Zerlegt einen Formel-String in Einzel-Tokens (Zahlen, Operatoren, Variablen)."""
    if not all(c.isalnum() or c in " +-*/^().," for c in expr):
        raise ValueError(f"Invalid character in expression: {expr}")
    pos, tokens = 0, []
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid token at position {pos}: '{expr[pos:]}'")
        token = m.group(1)
        pos = m.end()
        tokens.append(token)
    return tokens

# -------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------
class SymbolicRegressionEnv:
    def __init__(self, target_fn, allowed_operators=None, x_range=X_RANGE, n_points=N_POINTS, max_length=MAX_LENGTH):
        self.target_fn = target_fn
        self.x_range = x_range
        self.n_points = n_points
        self.max_length = max_length

        # Define tokens
        self.TOK_PAD   = 0   # Padding
        self.TOK_X     = 1   # Variable x
        self.TOK_CONST = 2   # Constants (will be selected from CONST_RANGE)
        self.TOK_PLUS  = 3   # Addition +
        self.TOK_MINUS = 4   # Subtraction -
        self.TOK_MUL   = 5   # Multiplication *
        self.TOK_DIV   = 6   # Division /
        self.TOK_POW   = 7   # Power ^
        self.TOK_SIN   = 8   # sin function
        self.TOK_COS   = 9   # cos function
        self.TOK_EXP   = 10  # exp function

        # Handle allowed operators
        self.allowed_operators = allowed_operators
        self.op_mapping = {
            '+': self.TOK_PLUS,
            '-': self.TOK_MINUS,
            '*': self.TOK_MUL,
            '/': self.TOK_DIV,
            '^': self.TOK_POW,
            'sin': self.TOK_SIN,
            'cos': self.TOK_COS,
            'exp': self.TOK_EXP
        }

        # Wenn allowed_operators angegeben, filtern wir die zulässigen Token
        if self.allowed_operators:
            self.binary_ops = [self.op_mapping[op] for op in self.allowed_operators
                            if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in self.allowed_operators
                            if op in ['sin', 'cos', 'exp']]
        else:
            # Define token properties (default)
            self.binary_ops = [self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW]
            self.unary_ops = [self.TOK_SIN, self.TOK_COS, self.TOK_EXP]

        self.terminals = [self.TOK_X, self.TOK_CONST]

        self.token_str = {
            self.TOK_X:     "x",
            self.TOK_CONST: "1",  # Placeholder, will be replaced with actual constant
            self.TOK_PLUS:  "+",
            self.TOK_MINUS: "-",
            self.TOK_MUL:   "*",
            self.TOK_DIV:   "/",
            self.TOK_POW:   "^",
            self.TOK_SIN:   "sin",
            self.TOK_COS:   "cos",
            self.TOK_EXP:   "exp",
        }

        # Vocabulary size based on available tokens
        self.vocab_size = len(self.token_str) + 1  # +1 for padding
        self.reset()

    def reset(self):
        self.tokens = []
        self.constants = []  # Liste zur Speicherung der Konstanten
        self.required_operands = 1
        self.steps = 0
        self.done = False
        self._need_variable_in_trig = False
        # Erzeugung der x-Werte
        X = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        self.x_values = X
        try:
            if callable(self.target_fn):
                # Hier werden die Zielwerte aus dem Interpolator berechnet
                self.target_y = torch.tensor([float(self.target_fn(x.item())) for x in X])
            else:
                self.target_y = self._eval_expr_string(self.target_fn, X)
        except Exception as e:
            print(f"Error evaluating target function: {e}")
            self.target_y = torch.zeros_like(X)
        self.prev_normalized_mse = 1.0
        return [self.TOK_PAD] * self.max_length


    def _eval_expr_string(self, expr_str, x_values):
        """Bewertet einen Ausdrucks-String elementweise für gegebene x-Werte (eval-safe)."""
        expr_str = expr_str.replace('^', '**')  # Handle power operator
        results = []
        for x in x_values:
            try:
                result = eval(expr_str,
                              {"__builtins__": {}},
                              {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': x.item()})
                results.append(float(result))
            except:
                results.append(float('nan'))
        return torch.tensor(results)

    def step(self, action_token):
        if self.done:
            raise RuntimeError("step() called after episode is done")

        self.steps += 1

        # Check if this is a nested function call that should be restricted
        if action_token in self.unary_ops:  # If it's a function (sin, cos, exp)
            # Check if we're already inside a function that restricts nesting
            if self._is_inside_restricted_function():
                # Invalid action - return negative reward
                self.done = True
                return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens)), -5.0, True, {'expr': None}

        # Special validation for ^ operator - only constants allowed as exponents
        if len(self.tokens) > 0 and self.tokens[-1] == self.TOK_POW:
            if action_token == self.TOK_X:  # Variable not allowed as exponent
                self.done = True
                return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens)), -5.0, True, {'expr': None}

        # Handle constants specially
        if action_token == self.TOK_CONST:
            # Select a constant value from predefined range
            const_value = random.choice(CONST_RANGE)
            self.constants.append(const_value)

        self.tokens.append(action_token)

        # Check for trig functions that have only constants (invalid)
        if len(self.tokens) >= 2:
            if (self.tokens[-2] in [self.TOK_SIN, self.TOK_COS] and
                self.tokens[-1] == self.TOK_CONST):
                # Need to make sure x will be used eventually in this function
                self._need_variable_in_trig = True

        # Update required operands based on token type
        if action_token in self.binary_ops:
            self.required_operands += 1
        elif action_token in self.unary_ops:
            pass  # Unary operators don't change the required operands
        else:
            self.required_operands -= 1

        reward, info = 0.0, {}

        # Check if expression is complete or max length is reached
        if self.required_operands == 0 or self.steps >= self.max_length:
            self.done = True
            if self.required_operands == 0:
                # Check for invalid trig expressions (sin/cos without variable)
                if self._has_invalid_trig_expressions():
                    reward = -5.0
                    info['expr'] = None
                else:
                    # Valid expression - calculate reward
                    reward = self._calculate_reward()
                    info['expr'] = self.get_expression_str()
            else:
                # Incomplete expression
                reward = -5.0
                info['expr'] = None

        # Return observation (padded token sequence)
        obs = self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))
        return obs, reward, self.done, info

    def _is_inside_restricted_function(self):
        """Prüft, ob aktuell in einer geschachtelten (verbotenen) Funktionsstruktur steckt."""
        stack = []

        for token in self.tokens:
            if token in self.unary_ops:  # sin, cos, exp
                stack.append(token)
            elif token in self.terminals:  # x or const
                if stack:  # If we're inside some function
                    # Check if terminal is a variable (needed for trig functions)
                    if token == self.TOK_X and stack[-1] in [self.TOK_SIN, self.TOK_COS]:
                        # Valid for trig functions - they have x
                        stack.pop()  # Function is complete
                    elif token == self.TOK_CONST and stack[-1] not in [self.TOK_SIN, self.TOK_COS]:
                        # Valid for non-trig functions
                        stack.pop()  # Function is complete
                    else:
                        pass  # Keep function open, waiting for a variable
                else:
                    pass  # Not inside a function, do nothing
            elif token in self.binary_ops:  # +, -, *, /
                pass  # Binary ops don't affect function nesting

        # If the stack is not empty, we're inside some function
        return len(stack) > 0

    def _has_invalid_trig_expressions(self):
        """Erkennt sin/cos-Aufrufe ohne Variablenanteil und meldet sie als ungültig."""
        # We'll use a simple stack-based approach to track function nesting
        stack = []
        trig_funcs = [self.TOK_SIN, self.TOK_COS]

        for token in self.tokens:
            if token in trig_funcs:
                stack.append((token, False))  # (function, has_variable)
            elif token == self.TOK_X and stack:
                # Mark the innermost function as having a variable
                func, _ = stack[-1]
                stack[-1] = (func, True)
            elif token in self.binary_ops:
                # For binary ops, we need to check if there's any incomplete trig function
                pass
            elif token in self.terminals and not token == self.TOK_X:
                # For constants, nothing changes for trig validation
                pass

        # If any trig function doesn't have a variable, it's invalid
        for func, has_var in stack:
            if func in trig_funcs and not has_var:
                return True

        return False

    def _calculate_reward(self):
        """Erzeugt einen Reward aus MSE + Bonus für relative Verbesserung."""
        try:
            y_pred = self._evaluate_expression()
        except Exception as e:
            return -5.0  # Fehlerhafte Ausdrücke bestrafen
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        # Form prüfen
        if y_pred.shape != self.target_y.shape:
            if y_pred.shape == torch.Size([]):
                y_pred = y_pred * torch.ones_like(self.target_y)
            else:
                return -5.0
        # Ungültige Werte abfangen
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            return -5.0

        # Berechne den MSE und normalisiere ihn mit der Varianz der Zielwerte
        mse = float(torch.mean((y_pred - self.target_y) ** 2))
        variance = float(torch.var(self.target_y)) + 1e-6  # Sicherheitsfaktor
        normalized_mse = mse / variance

        # Basisreward: exponentiell transformierter Fehler – typischerweise im Bereich (0,1]
        base_reward = math.exp(-normalized_mse)

        # Berechne den relativen Verbesserungsquotienten:
        # Falls self.prev_normalized_mse > 0, berechne den Quotienten, andernfalls ist kein Bonus möglich
        if self.prev_normalized_mse > 0:
            improvement_ratio = (self.prev_normalized_mse - normalized_mse) / self.prev_normalized_mse
        else:
            improvement_ratio = 0

        # Clippe den Quotienten zwischen 0 und 1 und skaliere ihn beispielsweise mit 0.1
        bonus =  BONUS_SCALE * max(0, min(improvement_ratio, 1))

        # Aktualisiere den gespeicherten Fehler für die nächste Bewertung
        self.prev_normalized_mse = normalized_mse

        # Gesamtreward ist Basisreward plus den kleinen Bonus
        reward = base_reward + bonus
        return reward


    def _evaluate_expression(self):
        """Parst und berechnet das aktuelle Präfix-Token-Array als Tensor-Ausgabe."""
        const_idx = 0  # Index to track which constant we're using

        def eval_prefix(idx):
            nonlocal const_idx
            if idx >= len(self.tokens):
                raise ValueError("Unexpected end of expression")

            tok = self.tokens[idx]

            # Handle binary operations
            if tok in self.binary_ops:
                left, next_idx = eval_prefix(idx + 1)
                right, next_idx = eval_prefix(next_idx)

                # Convert to tensors if needed
                if not isinstance(left, torch.Tensor): left = torch.tensor(left)
                if not isinstance(right, torch.Tensor): right = torch.tensor(right)

                # Perform operation
                if tok == self.TOK_DIV:
                    # Safe division
                    right_safe = torch.where(right == 0, torch.full_like(right, SAFE_DIV_EPS), right)
                    return left / right_safe, next_idx
                elif tok == self.TOK_PLUS:
                    return left + right, next_idx
                elif tok == self.TOK_MINUS:
                    return left - right, next_idx
                elif tok == self.TOK_POW:
                    # Safe power operation
                    # Limit base and exponent to avoid overflow
                    left_safe = torch.clamp(left, *BASE_CLAMP)

                    # Check if right is a constant (expected for power operations)
                    if not isinstance(right, torch.Tensor) or (isinstance(right, torch.Tensor) and right.numel() == 1):
                        right_safe = torch.clamp(right, *POW_CLAMP)  # Limit exponent range
                        # Handle negative bases with non-integer exponents
                        power_safe = torch.where(
                            (left_safe < 0) & (right_safe != right_safe.round()),
                            torch.abs(left_safe) ** right_safe,  # Use absolute value for base
                            left_safe ** right_safe
                        )
                        return power_safe, next_idx
                    else:
                        # Variable exponent - this should be restricted by the action selection
                        # Return a safe fallback value
                        return torch.ones_like(left), next_idx
                else:  # TOK_MUL
                    return left * right, next_idx

            # Handle unary operations
            elif tok in self.unary_ops:
                arg, next_idx = eval_prefix(idx + 1)
                if not isinstance(arg, torch.Tensor): arg = torch.tensor(arg)

                # For trig functions, ensure arg contains a variable component
                if tok in [self.TOK_SIN, self.TOK_COS]:
                    # This is a simplistic check - in practice, would need more robust analysis
                    # If arg is a constant tensor with same values, it's invalid
                    if arg.numel() > 1 and torch.all(arg == arg[0]):
                        # Invalid trig function without variable component
                        return torch.zeros_like(self.x_values), next_idx

                if tok == self.TOK_SIN:
                    return torch.sin(arg), next_idx
                elif tok == self.TOK_COS:
                    return torch.cos(arg), next_idx
                else:  # TOK_EXP
                    # Safe exp to prevent overflow
                    safe_arg = torch.clamp(arg, *EXP_CLAMP)  # Limit range to avoid overflow
                    return torch.exp(safe_arg), next_idx

            # Handle terminals
            elif tok == self.TOK_X:
                return self.x_values, idx + 1
            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return torch.full_like(self.x_values, val), idx + 1
                else:
                    # Fallback if constants are misaligned
                    return torch.ones_like(self.x_values), idx + 1

            raise ValueError(f"Unknown token {tok}")

        result, _ = eval_prefix(0)
        return result

    def get_expression_str(self):
        """Gibt die Token-Liste als lesbaren Infix-String zurück (falls vollständig)."""
        if self.required_operands != 0:
            return None

        const_idx = 0

        def build_expr(i):
            nonlocal const_idx
            tok = self.tokens[i]

            if tok in self.binary_ops:
                op = self.token_str[tok]
                left_expr, next_i = build_expr(i + 1)
                right_expr, next_i = build_expr(next_i)

                # Handle power operator with parentheses for clarity
                if tok == self.TOK_POW:
                    return f"({left_expr})^({right_expr})", next_i
                else:
                    return f"({left_expr} {op} {right_expr})", next_i

            elif tok in self.unary_ops:
                func = self.token_str[tok]
                arg_expr, next_i = build_expr(i + 1)
                return f"{func}({arg_expr})", next_i

            elif tok == self.TOK_X:
                return "x", i + 1

            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return str(val), i + 1
                else:
                    return "1", i + 1

            return "?", i + 1  # Fallback for unknown tokens

        expr_str, _ = build_expr(0)
        return expr_str


    def optimize_constants(self):
        """Feintuned numerische Konstanten per SciPy, um den Fehler weiter zu senken."""
        if not self.tokens or self.required_operands != 0:
            return False

        # Count how many constants are in the expression
        const_count = self.tokens.count(self.TOK_CONST)
        if const_count == 0:
            return False

        # Create a copy of current constants as starting point
        initial_constants = self.constants.copy()

        # Define error function for optimization
        def error_function(const_values):
            # Round constants to 0.01 precision
            rounded_values = [round(val * CONST_FACTOR) / CONST_FACTOR for val in const_values]

            # Temporarily replace constants
            self.constants = rounded_values
            try:
                # Calculate predictions with updated constants
                y_pred = self._evaluate_expression()

                # Check for invalid values
                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    return 1e10  # High error for invalid results

                # Calculate mean squared error
                mse = float(torch.mean((y_pred - self.target_y)**2))
                return mse
            except Exception:
                return 1e10  # High error value if exception occurs
            finally:
                # Restore original constants
                self.constants = initial_constants.copy()

        # Use multiple optimization methods with different starting points

        best_result = None
        best_mse = float('inf')

        # Try different starting points, rounded to 0.01
        starting_points = [
            [round(val * CONST_FACTOR) / CONST_FACTOR for val in initial_constants],
            [0.0] * const_count,
            [0.5] * const_count,
            [1.0] * const_count,
            [-0.5] * const_count,
            [-1.0] * const_count
        ] + [[round(random.uniform(-5, 5) * CONST_FACTOR) / CONST_FACTOR for _ in range(const_count)] for _ in range(3)]

        methods = ['L-BFGS-B', 'BFGS', 'Nelder-Mead']

        for start_point in starting_points:
            for method in methods:
                try:
                    bounds = None if method in ['BFGS', 'Nelder-Mead'] else [(-10, 10)] * const_count

                    result = optimize.minimize(
                        error_function,
                        start_point,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': MAXITER_OPT}
                    )

                    # Round result to 0.1 precision
                    if result.success:
                        rounded_result = [round(val * CONST_FACTOR) / CONST_FACTOR for val in result.x]
                        # Calculate MSE with rounded values
                        self.constants = rounded_result
                        try:
                            y_pred = self._evaluate_expression()
                            mse = float(torch.mean((y_pred - self.target_y)**2))
                            if mse < best_mse:
                                best_mse = mse
                                best_result = rounded_result
                        except:
                            pass
                        finally:
                            self.constants = initial_constants.copy()
                except Exception:
                    continue

        # Use best result if found
        if best_result is not None:
            self.constants = best_result
            return True

        return False


# ERSETZEN: Die komplette PolicyNetwork-Klasse

class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size - 1)  # -1 because we don't select PAD

        # Initialize weights
        for param in self.lstm.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, seq):
        """
        Berechnet Aktions-Logits für einen Batch von Token-Sequenzen.
        Diese Methode verarbeitet jetzt korrekt einzelne Sequenzen und Batches.
        """
        # Sicherstellen, dass der Input ein LongTensor ist.
        if not isinstance(seq, torch.Tensor):
            seq = torch.LongTensor(seq)
        # Wenn nur eine Sequenz (z.B. bei Aktionsauswahl) übergeben wird,
        # fügen wir eine Batch-Dimension hinzu.
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)

        # Embedding -> LSTM
        # output shape: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.lstm(self.embed(seq))

        # Wir nehmen den letzten Hidden State jeder Sequenz im Batch.
        # output[:, -1, :] hat die Form (batch_size, hidden_size)
        last_hidden_state = output[:, -1, :]

        # Linear Layer zur Berechnung der Logits
        # output shape: (batch_size, vocab_size - 1)
        logits = self.fc(last_hidden_state)
        
        return logits

    def select_action(self, seq, valid_actions=None):
        """Wählt nach Policy oder Zufall eine gültige nächste Aktion aus."""
        with torch.no_grad():
            # Die forward-Methode gibt jetzt immer einen 2D-Tensor zurück,
            # daher .squeeze(0) für den Fall einer einzelnen Aktion.
            logits = self.forward(seq).squeeze(0)

            # Apply mask if valid_actions are provided
            if valid_actions is not None:
                mask = torch.full_like(logits, -float('inf'))
                mask[valid_actions] = 0
                masked_logits = logits + mask
                dist = torch.distributions.Categorical(logits=masked_logits)
            else:
                dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            return action.item() + 1, dist.log_prob(action), dist.entropy()


# -------------------------------------------------
# REPLAY BUFFER
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# -------------------------------------------------
# TRAINER
# -------------------------------------------------
# ERSETZEN: Die komplette DSPTrainer-Klasse
class DSPTrainer:
    def __init__(self, target_fn, allowed_operators=None, episodes=EPISODES, lr=LR,
                 entropy_coef=ENTROPY_COEF, batch_size=BATCH_SIZE):
        self.env = SymbolicRegressionEnv(target_fn, allowed_operators=allowed_operators)
        self.model = PolicyNetwork(self.env.vocab_size)
        self.target_model = PolicyNetwork(self.env.vocab_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.episodes = episodes
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.steps_done = 0
        
        # Erweiterte Attribute zum Sammeln von Daten für die Analyse
        self.history = []
        self.entropy_history = []
        self.action_probs_history = []
        self.prob_log_epochs = []
        self.top_solutions = []
        self.seen_expressions = set()
        
        # Frequenz für das Loggen der Wahrscheinlichkeiten
        self.PROB_LOG_FREQ = 250

    def get_valid_actions(self, required_operands):
        """Liefert zulässige Token-IDs basierend auf der aktuellen Ausdrucksstruktur."""
        valid_actions = []
        if required_operands > 0:
            if not self.env._is_inside_restricted_function():
                valid_actions.extend(self.env.unary_ops)
            valid_actions.extend(self.env.binary_ops)
            if len(self.env.tokens) == 0 or self.env.tokens[-1] != self.env.TOK_POW:
                valid_actions.extend(self.env.terminals)
            else:
                valid_actions.append(self.env.TOK_CONST)
        return list(set([a - 1 for a in valid_actions])) # Indizes für das Netzwerk

    def select_action(self, state, required_operands):
        """Epsilon-greedy Wahl einer Aktion."""
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        valid_actions = self.get_valid_actions(required_operands)
        if not valid_actions: return None, None, None

        if random.random() < epsilon:
            action = random.choice(valid_actions) + 1
            return action, None, None
        else:
            return self.model.select_action(state, valid_actions)

    def optimize_model(self):
        """Aktualisiert die Policy-Gewichte mithilfe von Replay-Samples (DQN-ähnlich)."""
        if len(self.memory) < self.batch_size: return
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

        state_tensor = torch.LongTensor(state_batch)
        action_tensor = torch.LongTensor([a - 1 for a in action_batch]).unsqueeze(1)
        reward_tensor = torch.FloatTensor(reward_batch)
        next_state_tensor = torch.LongTensor(next_state_batch)
        done_tensor = torch.FloatTensor(done_batch)

        current_q_values = self.model(state_tensor).gather(1, action_tensor)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_tensor).max(1)[0]
            target_q_values = (reward_tensor + GAMMA * next_q_values * (1 - done_tensor)).unsqueeze(1)
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _update_top_solutions(self, reward, expr):
        """Hält eine Liste der Top-3-Lösungen aktuell."""
        if expr is None or expr in self.seen_expressions: return
        self.seen_expressions.add(expr)
        if len(self.top_solutions) < 3:
            self.top_solutions.append((reward, expr))
        elif reward > self.top_solutions[-1][0]:
            self.top_solutions[-1] = (reward, expr)
        self.top_solutions.sort(key=lambda x: x[0], reverse=True)

    def simplify_expression(self, expr_str):
        if not expr_str: return None
        try:
            x = sp.Symbol('x')
            expr_py = expr_str.replace('^', '**')
            expr = sp.sympify(expr_py, locals={'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
            return str(sp.expand(expr)).replace('**', '^')
        except Exception: return expr_str

    def train(self):
        """Führt das RL-Training aus und sammelt Daten für die Analyse."""
        for episode in enhanced_progress_bar(range(1, self.episodes + 1), desc="SR+RL Training", unit=" Ep."):
            state = self.env.reset()
            episode_reward, entropies = 0.0, []
            
            while not self.env.done:
                action, log_prob, entropy = self.select_action(state, self.env.required_operands)
                if action is None: break
                if entropy is not None: entropies.append(entropy.item())

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.memory.add(state, action, next_state, reward, done)
                state = next_state

            if self.env.done and info.get('expr') and self.env.tokens.count(self.env.TOK_CONST) > 0:
                if self.env.optimize_constants():
                    episode_reward = self.env._calculate_reward()
                    info['expr'] = self.env.get_expression_str()

            if info.get('expr'):
                simplified_expr = self.simplify_expression(info.get('expr'))
                self._update_top_solutions(episode_reward, simplified_expr)

            self.history.append(episode_reward)
            if entropies: self.entropy_history.append(np.mean(entropies))

            if episode % POLICY_UPDATE_FREQ == 0: self.optimize_model()
            if episode % TARGET_UPDATE_FREQ == 0: self.update_target_network()
            
            # Loggen der Aktionswahrscheinlichkeiten
            if episode % self.PROB_LOG_FREQ == 0:
                self.prob_log_epochs.append(episode)
                with torch.no_grad():
                    logits = self.model([self.env.TOK_PAD] * self.env.max_length)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    self.action_probs_history.append(probs)

        return self.top_solutions

# -------------------------------------------------
# WRAPPER FUNCTIONS
# -------------------------------------------------
# ERSETZEN: Die optimize_expression Funktion
def optimize_and_analyze_expression(target_fn, allowed_operators, episodes, run_name):
    """
    Startet den DSPTrainer, führt die Analyse-Plots aus und liefert die besten
    gefundenen symbolischen Ausdrücke zurück.
    """
    trainer = DSPTrainer(target_fn, allowed_operators=allowed_operators, episodes=episodes)
    top_solutions = trainer.train()

    # Nach dem Training die Analyse-Plots für diesen Lauf anzeigen
    print_highlighted(f"\n--- Analysediagramme für Lauf: {run_name} ---", Fore.YELLOW)
    plot_reward_history(trainer.history, run_name)
    plot_entropy_history(trainer.entropy_history, run_name)
    plot_action_prob_heatmap(trainer.action_probs_history, trainer.prob_log_epochs, trainer.env, run_name)
    
    return top_solutions

def calculate_mse(target_fn, expr_str, x_range=(0, L), n_points=200):
    """Berechnet den MSE zwischen Ziel- und Symbolic-Funktion auf gemeinsamem Gitter."""
    if not expr_str:
        return float('inf')
    try:
        x = torch.linspace(x_range[0], x_range[1], n_points)
        
        # Evaluate target function (PINN Interpolator)
        if callable(target_fn):
            y_true = torch.tensor([float(target_fn(xi.item())) for xi in x], dtype=torch.float32)
        else: # String expression
            # Diese Auswertung für y_true könnte ebenfalls angepasst werden, ist aber für Ihr aktuelles Problem nicht die Ursache.
            expr_true = str(target_fn).replace('^', '**')
            y_true = torch.tensor([eval(expr_true, {"__builtins__": {}}, {**SAFE_FUNCS, 'x': xi.item()}) for xi in x], dtype=torch.float32)

        # === KORREKTUR HIER ===
        # Erstellen Sie eine sichere Umgebung für eval() mit den Standard-MATH-Funktionen.
        safe_math_dict = {
            'sin': math.sin,
            'cos': math.cos,
            'exp': math.exp,
            # Füge weitere bei Bedarf hinzu
        }

        # Evaluate approximated expression
        expr_pred = expr_str.replace('^', '**')
        
        # Werten Sie den Ausdruck für jeden Punkt mit der math-Umgebung aus
        y_pred_vals = []
        for xi in x:
            safe_math_dict['x'] = xi.item() # Fügt den aktuellen x-Wert hinzu
            y_pred_vals.append(eval(expr_pred, {"__builtins__": {}}, safe_math_dict))
            
        y_pred = torch.tensor(y_pred_vals, dtype=torch.float32)
        # === ENDE DER KORREKTUR ===

        # Filter out invalid values
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y_true) | torch.isinf(y_true))
        if valid_mask.sum() == 0:
            return float('inf')

        mse = torch.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2).item()
        return mse
    except Exception as e:
        # print(f"Error calculating MSE for '{expr_str}': {e}") # Zur Fehlersuche einkommentieren
        return float('inf')
# HINZUFÜGEN: Import für die Heatmap
import seaborn as sns

# HINZUFÜGEN: Neue Plotting-Funktionen
def moving_average(data, window_size):
    """Berechnet den gleitenden Durchschnitt."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_reward_history(history, run_name):
    """Plottet den Reward-Verlauf mit gleitendem Durchschnitt."""
    plt.figure(figsize=(12, 6))
    plt.plot(history, label='Reward pro Episode', alpha=0.5)
    if len(history) > 100:
        avg = moving_average(history, 100)
        plt.plot(np.arange(99, len(history)), avg, color='red', linewidth=2, label='Gleitender Durchschnitt (100 Ep.)')
    plt.title(f'Reward-Verlauf - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Gesamt-Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_entropy_history(entropy_history, run_name):
    """Plottet den Verlauf der Policy-Entropie."""
    plt.figure(figsize=(12, 6))
    plt.plot(entropy_history, label='Durchschnittliche Entropie pro Episode', alpha=0.6)
    if len(entropy_history) > 100:
        avg = moving_average(entropy_history, 100)
        plt.plot(np.arange(99, len(entropy_history)), avg, color='green', linewidth=2, label='Gleitender Durchschnitt (100 Ep.)')
    plt.title(f'Policy-Entropie-Verlauf - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Entropie')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_action_prob_heatmap(action_probs, epochs, env, run_name):
    """Stellt die Aktionswahrscheinlichkeiten als Heatmap dar, OHNE die Konstante '1'."""
    if not action_probs: return

    # 1. Die originalen Labels, die den Netzwerk-Ausgaben entsprechen
    #    (Index 0 -> Token 1, Index 1 -> Token 2, etc.)
    original_labels = [env.token_str.get(i + 1, f"UNK_{i+1}") for i in range(env.vocab_size - 1)]
    
    # 2. Konvertiere die rohen Wahrscheinlichkeitsdaten in ein NumPy-Array
    probs_array = np.array(action_probs)

    # 3. Finde den Index für die Konstante '1' (TOK_CONST = 2, also Index = 1)
    const_index = env.TOK_CONST - 1

    # 4. Entferne die Spalte und das Label an diesem Index
    if const_index < len(original_labels):
        # Entferne die Spalte aus den Daten
        filtered_probs = np.delete(probs_array, const_index, axis=1)
        # Entferne das Label aus der Liste
        filtered_labels = [label for i, label in enumerate(original_labels) if i != const_index]
    else:
        # Fallback, falls etwas nicht stimmt
        filtered_probs = probs_array
        filtered_labels = original_labels

    # 5. Plotte die GEFILTERTEN Daten und Labels
    plt.figure(figsize=(14, 8))
    sns.heatmap(filtered_probs, cmap='viridis', xticklabels=filtered_labels, yticklabels=epochs)
    plt.title(f'Heatmap der Aktionswahrscheinlichkeiten - {run_name} (ohne Konstante)')
    plt.xlabel('Aktion (Token)')
    plt.ylabel('Episode')
    plt.xticks(rotation=45)
    plt.show()

def plot_single_result(target_fn, expr_str, title):
    """
    Plottet das Ergebnis eines einzelnen Laufs (PINN-Ziel vs. gefundene Formel).
    """
    plt.figure(figsize=(10, 6))
    x_plot = torch.linspace(0, L, 200)

    # PINN-Zieldaten (y_true)
    y_true = [float(target_fn(xi.item())) for xi in x_plot]
    plt.plot(x_plot.numpy(), y_true, 'k-', label="PINN-Ziel", linewidth=2.5)

    # Gefundene Formel (y_pred)
    if expr_str and expr_str != "Fehlgeschlagen":
        try:
            # Sichere Umgebung für eval()
            safe_math_dict = {'sin': math.sin, 'cos': math.cos, 'exp': math.exp, 'x': 0}
            expr_py = expr_str.replace('^', '**')

            y_pred_vals = []
            for xi in x_plot:
                safe_math_dict['x'] = xi.item()
                y_pred_vals.append(eval(expr_py, {"__builtins__": {}}, safe_math_dict))

            plt.plot(x_plot.numpy(), y_pred_vals, 'r--', label=f"SR-Formel", linewidth=2)
        except Exception as e:
            print(f"Konnte Ausdruck für Plot nicht auswerten: {expr_str}, Fehler: {e}")

    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("Durchbiegung u(x) [m]")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------
# =================================================================================
# ERSETZEN SIE IHREN GESAMTEN MAIN-BLOCK DURCH DIESEN HIER
# =================================================================================
if __name__ == "__main__":
    # === PHASE 1: PINN SIMULATION ===
    run_pinn_simulation()

    # --- Laden und Plotten der PINN-Ergebnisse ---
    x_nn, u_nn = np.loadtxt("PINNBeamForward_py.txt", unpack=True)
    x_an, u_an = np.loadtxt("LSGBeamForward_py.txt", unpack=True)
    epoch, zf_error = np.loadtxt("ErrorFileBeamForward_py.txt", unpack=True)

    plt.figure(figsize=(10, 5))
    plt.plot(x_an, u_an, 'k--', label="Analytische Lösung", linewidth=2)
    plt.plot(x_nn, u_nn, 'r-',  label="PINN-Vorhersage", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Durchbiegung u(x) [m]")
    plt.title("Phase 1: PINN vs. Analytische Lösung")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch, zf_error, 'b-')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Zielfunktionswert (Loss)")
    plt.title("Phase 1: Konvergenzverlauf des PINN")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === PHASE 2: SYMBOLIC REGRESSION (SR+RL) ===
    print_highlighted("\nPhase 2: Starte Symbolische Regression mit Reinforcement Learning...", Fore.CYAN)
    
    # Erstelle die Interpolationsfunktion aus den PINN-Daten
    target_fn = interp1d(x_nn.flatten(), u_nn.flatten(),
                         kind='cubic', fill_value="extrapolate")

    # Definiere die verschiedenen Testläufe mit unterschiedlichen Operatoren
    TARGET_FUNCTIONS = [
        ("PINN-Approximation (alle Operatoren)", target_fn, ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("PINN-Approximation (ohne exp)", target_fn, ['+', '-', '*', '^', 'sin', 'cos']),
        ("PINN-Approximation (Polynom)", target_fn, ['+', '-', '*', '^']),
        ("PINN-Approximation (linear)", target_fn, ['+', '-', '*']),
    ]

    results = {}
    plot_data = []

    for i, (name, fn, operators) in enumerate(TARGET_FUNCTIONS):
        print_highlighted(f"\n=== Starte SR+RL Lauf {i+1}/4: {name} ===", Fore.YELLOW)
        
        top_solutions = optimize_and_analyze_expression(fn, allowed_operators=operators, episodes=EPISODES, run_name=name)
        
        if top_solutions:
            best_reward, best_expr = top_solutions[0]
            mse = calculate_mse(fn, best_expr)

            print_highlighted(f"→ Bester gefundener Ausdruck: {best_expr}", Fore.LIGHTGREEN_EX)
            print(f"→ Reward: {best_reward:.4f}")
            print(f"→ MSE: {mse:.6g}")

            # HIER WIRD DER NEUE PLOT FÜR DAS EINZELERGEBNIS AUFGERUFEN
            plot_single_result(fn, best_expr, f"Ergebnis für: {name}")

            run_name_key = f"Lauf {i+1}"
            results[run_name_key] = (best_expr, best_reward, mse, ' '.join(operators))
            plot_data.append((fn, best_expr, name, mse))
        else:
            print_highlighted(f"→ Für '{name}' wurde keine gültige Lösung gefunden.", Fore.RED)
            run_name_key = f"Lauf {i+1}"
            results[run_name_key] = ("Fehlgeschlagen", 0, float('inf'), ' '.join(operators))

    # --- Zusammenfassung der SR+RL Ergebnisse in der Konsole ---
    print_highlighted("\n\n=== Zusammenfassung der Symbolischen Regression ===", Fore.CYAN)
    table_data = []
    headers = ["Lauf", "Operatoren", "Bester Ausdruck", "Reward", "MSE"]
    for name, (expr, reward, mse, ops) in results.items():
        table_data.append([name, ops, expr if expr else "Fehlgeschlagen", f"{reward:.3f}", f"{mse:.6g}"])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # --- Kombinierter finaler Plot aller SR+RL Ergebnisse ---
    plt.figure(figsize=(15, 12))
    plt.suptitle("Finaler Vergleich: Beste SR-Formeln vs. PINN-Ziel", fontsize=16)

    for i, (target_fn, expr_str, run_name, mse) in enumerate(plot_data, 1):
        ax = plt.subplot(2, 2, i)
        x_plot = torch.linspace(0, L, 200)
        y_true = [float(target_fn(xi.item())) for xi in x_plot]

        ax.plot(x_plot.numpy(), y_true, '-', color='black', label="PINN-Ziel", linewidth=2.5)

        if expr_str:
            try:
                safe_math_dict = {'sin': math.sin, 'cos': math.cos, 'exp': math.exp, 'x': 0}
                expr_py = expr_str.replace('^', '**')
                
                y_pred_vals = []
                for xi in x_plot:
                    safe_math_dict['x'] = xi.item()
                    y_pred_vals.append(eval(expr_py, {"__builtins__": {}}, safe_math_dict))
                
                ax.plot(x_plot.numpy(), y_pred_vals, '--', color='red', label=f"SR-Formel", linewidth=2)
            except Exception:
                pass # Fehlerhafte Ausdrücke nicht plotten

        ax.set_title(f"{run_name}\n MSE: {mse:.6g}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("u(x) [m]")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- Ergebnisse in Datei speichern ---
    with open('symbolic_regression_results.txt', 'w') as f:
        f.write("Ergebnisse der Symbolischen Regression\n")
        f.write("=======================================\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    print_highlighted("\nAlle Ergebnisse wurden in 'symbolic_regression_results.txt' gespeichert.", Fore.CYAN)

    # Alle am Ende gesammelten Plots anzeigen
    plt.show()