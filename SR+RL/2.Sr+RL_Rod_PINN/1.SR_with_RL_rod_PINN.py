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
from scipy import optimize

sys.setrecursionlimit(10000)



# -------------------------------------------------
# Problem­parameter DGL
# -------------------------------------------------
E  = 100.0   # Elastizitätsmodul
A  = 1.0     # Querschnittsfläche
q0 = 10.0    # Belastungskonstante
L  = 5.0     # Stablänge

# -------------------------------------------------
# Wichtige Hyperparameter PINN
# -------------------------------------------------
N_f         = 100          # Anzahl Kollokationspunkte (interne Stützstellen für PDE-Residual)
layers      = [1, 4, 8, 12, 8, 4, 1]   # Netz­architektur, Schichtgrößen des voll­verbundenen PINN
learning_rate = 1e-3       # Lernrate des PINN-Adam-Optimierers
n_epochs      = 10000     # Trainings­epochen für das PINN
ACTIVATION_FN = nn.Tanh() # Aktivierungsfunktion aller Hidden-Layer (z. B. nn.Tanh())




# -------------------------------------------------
# Daten­punkte / Kollokations­stellen
# -------------------------------------------------
x_f = torch.linspace(0, L, N_f, requires_grad=True).view(-1, 1)

# -------------------------------------------------
# PINN-Modell
# -------------------------------------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = ACTIVATION_FN
        
    def forward(self, x):
        # feedforward: In allen Schichten außer der letzten soll die Aktivierungsfunktion
        # angewandt werden.
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        # Letzte Schicht – keine Aktivierung, da wir die Rohwerte (u(x)) benötigen
        x = self.layers[-1](x)
        return x

# Modell instanziieren
model = PINN(layers)

# -------------------------------------------------
# Physik-Residual der PDE
# -------------------------------------------------
def pde_residual(model, x):
    """
    Berechnet den PDE-Residual:
        R(x) = AE * u''(x) + q0*x,
    sodass bei perfekter Lösung R(x) == 0.
    """
    # u(x) wird durch das Netz approximiert:
    u = model(x)
    
    # Berechne die erste Ableitung du/dx
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Berechne die zweite Ableitung d²u/dx²
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # PDE Residual, sodass der Ausdruck im Idealfall 0 ergeben soll:
    # AE * u_xx + q0*x = 0.
    residual = A * E * u_xx + q0 * x
    return residual

# -------------------------------------------------
# Verlustfunktion (Physik + Randbedingungen)
# -------------------------------------------------
def loss_function(model, x_f):
    """
    Addiert PDE-Fehler und Randbedingungen zu einem Gesamtverlust.
    """
    # PDE-Verlust über alle Kollokationspunkte
    f = pde_residual(model, x_f)
    loss_f = torch.mean(f**2)
    
    # Randbedingung an x = 0 : u(0) = 0
    x0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = model(x0)
    loss_bc1 = u0**2  # Quadratischer Fehler, es wird (u(0)-0)²
    
    # Randbedingung an x = L : u'(L) = 0
    xL = torch.tensor([[L]], requires_grad=True)
    uL = model(xL)
    uL_x = torch.autograd.grad(uL, xL, grad_outputs=torch.ones_like(uL), create_graph=True)[0]
    loss_bc2 = uL_x**2  # (u'(L)-0)²
    
    # Gesamter Verlust: Summe aller Fehlerterme
    loss = loss_f + loss_bc1 + loss_bc2
    return loss

# -------------------------------------------------
# Training
# -------------------------------------------------
# Adam Optimierer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Liste zur Speicherung der Verlustwerte (zum Plotten)
loss_history = []


for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    loss = loss_function(model, x_f)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Epoche {epoch}: Verlust = {loss.item():.6e}")

# -------------------------------------------------
# Visualisierung der Lösung
# -------------------------------------------------
# Erstelle Tensore für Visualisierung (fein aufgelöste Punkte im Intervall)
x_val = torch.linspace(0, L, 200).view(-1,1)
u_pred = model(x_val).detach().numpy()  # Netzvorhersage

plt.figure(figsize=(8,5))
plt.plot(x_val.numpy(), u_pred, label='PINN Output', linewidth=2)
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Approximierte Verschiebung $u(x)$')
plt.legend()
plt.show()

# -------------------------------------------------
# Dichte Auswertung + erneuter Plot
# -------------------------------------------------
# Erzeuge dichte Datenpunkte (x, u(x)) mit deinem trainierten Modell
x_dense = torch.linspace(0, L, 200).view(-1, 1)
u_dense = model(x_dense).detach().numpy()  # u(x) als Numpy-Array

# Plot zur Kontrolle
plt.figure(figsize=(8, 5))
plt.plot(x_dense.numpy(), u_dense, label="PINN Lösung", linewidth=2)
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.title("Numerische Lösung (PINN)")
plt.legend()
plt.show()

# -------------------------------------------------
# Interpolations­funktion für beliebige x
# -------------------------------------------------
# Erstelle einen Interpolator (stetige Funktion), der für ein gegebenes x den
# interpolierten u-Wert zurückgibt.
target_fn = interp1d(x_dense.numpy().flatten(), u_dense.flatten(), 
                     kind='cubic', fill_value="extrapolate")

# Teste den Interpolator an einem Beispielwert
print("Interpolierter Wert bei x=0.5:", target_fn(0.5))




# -------------------------------------------------
# Hyperparamter für Symbolic Policy with Reinforcement Learning
# -------------------------------------------------
MAX_LENGTH    = 12      # Maximale Token-Länge eines Ausdrucks
X_RANGE       = (0, L)  #Betrachtung des Inetrvalls, muss mit L übereinstimmen
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
EXP_CLAMP     =  (-20, 20)  # Overflow-Schwelle für exp-Aufruf, Wertebereich, auf den Argumente von exp() begrenzt werden
BASE_CLAMP    = (-100, 100) # Overflow / NaN-Schutz, für Bsis Exponent, Wertebereich der Basis bei Potenz-Operation
POW_CLAMP      = (-5, 5)    # Begrenzung Exponent in ^, erlaubter Exponenten-Bereich
GRAD_CLIP      =1.0         # Gradienren_Clipping, um zu große Updates zu veremiden, wenn größer oder kleiner -1, dann auf max 1 oder -1 gesetzt,Komponentenweises Gradient-Clipping (±GRAD_CLIP)
BONUS_SCALE    =0.1         # Einfluss der Verbesserungs-Prämie, Gewicht des relativen Verbesserungs-Bonus im Reward

# ------- Konstantenoptimierung --------------------------------------
CONST_DECIMALS = 2                     #Rundungsgenauigkeit (2 → 0.01-Raster), # 0 = ganze Zahl, 1 = Zehntel 0.1 , 2 = Hundertstel 0.01 ...
CONST_FACTOR   = 10 ** CONST_DECIMALS  # Hilfsfaktor zum Runden
CONST_RANGE = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.14, -3.14, 0.25, -0.25, 5.0, -5.0, 10.0, -10.0, math.pi, math.e] # kann man beliebig anpassen, Liste zulässiger Konstanten, wenn der zu Beginn der Optimierung der Agent „TOK_CONST“ wählt
MAXITER_OPT    =100                    #max. Iterationen für SciPy-minimize
                
                


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
                    left_safe = torch.clamp(left, BASE_CLAMP)
                    
                    # Check if right is a constant (expected for power operations)
                    if not isinstance(right, torch.Tensor) or (isinstance(right, torch.Tensor) and right.numel() == 1):
                        right_safe = torch.clamp(right, POW_CLAMP)  # Limit exponent range
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
                    safe_arg = torch.clamp(arg, EXP_CLAMP)  # Limit range to avoid overflow
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
    

# -------------------------------------------------
# POLICY NETWORK
# -------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size - 1)  # -1 because we don't select PAD
        
        # Initialize weights
        nn.init.zeros_(self.embed.weight[0])  # Zero for padding
        for param in self.lstm.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, seq):
        """Berechnet Aktions-Logits für eine Token-Sequenz (Embedding → LSTM → Linear)."""
        # Behandle verschiedene Eingabeformate
        if isinstance(seq, torch.Tensor):
            if seq.dim() > 2:  # Wenn es ein Batch mit mehr als 2 Dimensionen ist
                batch_size = seq.size(0)
                # Reshape auf 2D für LSTM
                reshaped = seq.reshape(batch_size, -1)
                embedded = self.embed(reshaped)
                output, (hidden, _) = self.lstm(embedded)
                return self.fc(hidden.squeeze(0))
            else:
                t = seq
        else:
            # Originaler Code für Listen-Eingabe
            t = torch.LongTensor(seq).unsqueeze(0)
        
        seq_list = t.tolist()[0] if t.dim() > 1 else t.tolist()
        
        # Find sequence length (up to padding or end)
        seq_len = seq_list.index(0) if 0 in seq_list else len(seq_list)
        
        if seq_len == 0:
            # Handle empty sequence
            h = torch.zeros((1, self.lstm.hidden_size))
        else:
            # Process through LSTM
            embedded = self.embed(t[:, :seq_len] if t.dim() > 1 else t[:seq_len])
            output, (hidden, _) = self.lstm(embedded)
            h = output[:, -1, :] if output.dim() > 2 else output[:, -1]
        
        # Final logits
        return self.fc(h).squeeze(0)
    
    def select_action(self, seq, valid_actions=None):
        """Wählt nach Policy oder Zufall eine gültige nächste Aktion aus."""
        with torch.no_grad():
            logits = self.forward(seq)
            
            # Apply mask if valid_actions are provided
            if valid_actions is not None:
                mask = torch.zeros_like(logits)
                mask[valid_actions] = 1
                masked_logits = logits + (mask - 1) * 1e9  # Large negative value for invalid actions
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
class DSPTrainer:
    def __init__(self, target_fn, allowed_operators=None, episodes=EPISODES, lr=LR, 
                entropy_coef=ENTROPY_COEF, batch_size=BATCH_SIZE):
        self.env = SymbolicRegressionEnv(target_fn, allowed_operators=allowed_operators)
    # Rest bleibt gleich...
        self.model = PolicyNetwork(self.env.vocab_size)
        self.target_model = PolicyNetwork(self.env.vocab_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
        self.episodes = episodes
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.steps_done = 0
        self.baseline = None
        self.history = []
        self.best_expr = None
        self.best_reward = float('-inf')
    
    def get_valid_actions(self, required_operands):
        """Liefert zulässige Token-IDs basierend auf der aktuellen Ausdrucksstruktur."""
        valid_actions = []
        
        # Always allow binary operations if we have at least one operand
        if required_operands > 0:
            valid_actions.extend(self.env.binary_ops)
            
            # Only add unary operators if not inside a restricted function
            if not self.env._is_inside_restricted_function():
                valid_actions.extend(self.env.unary_ops)
        
        # Allow terminals if we need operands
        if required_operands > 0:
            # If the last token is ^ (power), only allow constants as exponents
            if len(self.env.tokens) > 0 and self.env.tokens[-1] == self.env.TOK_POW:
                valid_actions.append(self.env.TOK_CONST)  # Only constants allowed as exponents
            else:
                valid_actions.extend(self.env.terminals)
        
        # Convert to indices for the PolicyNetwork (subtract 1 because tokens start at 1)
        return [a - 1 for a in valid_actions]
    
    def select_action(self, state, required_operands):
        """Epsilon-greedy Wahl einer Aktion samt Log-Wahrscheinlichkeit und Entropie."""
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        
        # Get valid actions
        valid_actions = self.get_valid_actions(required_operands)
        
        if random.random() < epsilon:
            # Random action
            action = random.choice(valid_actions) + 1
            return action, None, None
        else:
            # Policy-based action
            return self.model.select_action(state, valid_actions)
    
    def optimize_model(self):
        """Aktualisiert die Policy-Gewichte mithilfe von Replay-Samples (DQN-ähnlich)."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)
        
        # Convert to tensors with correct format
        state_tensor = torch.LongTensor(state_batch)
        next_state_tensor = torch.LongTensor(next_state_batch)
        action_tensor = torch.LongTensor([a - 1 for a in action_batch])  # Adjusted - no need for extra dimension
        reward_tensor = torch.FloatTensor(reward_batch)
        done_tensor = torch.FloatTensor(done_batch)
        
        # Current Q values - ensure proper shape for gather operation
        logits = self.model(state_tensor)
        # Check if logits are 1D or 2D
        if logits.dim() == 1:
            # If 1D, we need to handle a single sample case
            current_q_values = logits[action_tensor]
        else:
            # If 2D (batch, actions), use gather on dim 1
            current_q_values = logits.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # Compute target Q values with target network
        with torch.no_grad():
            next_logits = self.target_model(next_state_tensor)
            if next_logits.dim() == 1:
                next_q_values = next_logits.max()
            else:
                next_q_values = next_logits.max(1)[0]
            target_q_values = reward_tensor + GAMMA * next_q_values * (1 - done_tensor)
            
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping with check for None gradients
        for param in self.model.parameters():
            if param.grad is not None:  # Check if gradient exists
                param.grad.data.clamp_(-GRAD_CLIP, GRAD_CLIP)
                
        self.optimizer.step()
        
        return loss.item()

    
    def update_target_network(self):
        """Kopiert Gewichte vom Policy- auf das Target-Netz für stabilere Q-Schätzungen."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        """Führt das komplette RL-Training über alle Episoden hinweg aus."""
        for episode in enhanced_progress_bar(range(1, self.episodes + 1), desc="Training läuft", unit=" Ep."):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0

            # Episode loop
            while not self.env.done:
                # Select action
                action, log_prob, entropy = self.select_action(state, self.env.required_operands)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward

                # Store in replay buffer
                self.memory.add(state, action, next_state, reward, done)

                # Move to next state
                state = next_state

            # Optimize constants for completed expressions with constants
            if self.env.done and info.get('expr') and self.env.tokens.count(self.env.TOK_CONST) > 0:
                # Try multiple optimizations with different starting values
                best_reward = episode_reward
                best_constants = self.env.constants.copy()

                # Try different starting values
                for _ in range(3):  # Try a few different initializations
                    # Reset to random values
                    self.env.constants = [random.uniform(-5, 5) for _ in range(len(self.env.constants))]
                    if self.env.optimize_constants():
                        new_reward = self.env._calculate_reward()
                        if new_reward > best_reward:
                            best_reward = new_reward
                            best_constants = self.env.constants.copy()

                # Use the best constants found
                self.env.constants = best_constants
                episode_reward = best_reward
                info['expr'] = self.env.get_expression_str()

            # Update best expression if better
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_expr = info.get('expr')
                
                # Ausmultiplizieren des besten Ausdrucks
                if self.best_expr:
                    simplified_expr = self.simplify_expression(self.best_expr)
                    if simplified_expr:  # Falls die Vereinfachung erfolgreich war
                        self.best_expr = simplified_expr
                        
                if self.best_expr:
                    if self.best_expr:
                        print_highlighted(f"Episode {episode}/{self.episodes} – Neuer bester Ausdruck: {self.best_expr} | Reward: {self.best_reward:.3f}", Fore.GREEN)


            # Store episode reward
            self.history.append(episode_reward)

            # Update baseline reward
            if self.baseline is None:
                self.baseline = episode_reward
            else:
                self.baseline = 0.95 * self.baseline + 0.05 * episode_reward

            # Optimize model
            if episode % POLICY_UPDATE_FREQ == 0 and len(self.memory) >= self.batch_size:
                loss = self.optimize_model()

            # Update target network
            if episode % TARGET_UPDATE_FREQ  == 0:
                self.update_target_network()

            # Log progress periodically
            if episode % 100 == 0:
                avg_reward = sum(self.history[-100:]) / min(100, len(self.history[-100:]))
                print(f"Episode {episode}/{self.episodes}, Avg reward: {avg_reward:.3f}, Best: {self.best_expr}")

        self.optimize_best_expression()
        return self.best_expr, self.best_reward

    def get_final_expression(self):
        """Erzeugt aus der gelernten Policy einen deterministischen Best-Guess-Ausdruck."""
        self.env.reset()
        tokens = []
        required_operands = 1
        const_idx = 0
        
        while required_operands > 0 and len(tokens) < self.env.max_length:
            # Get valid actions
            valid_actions = self.get_valid_actions(required_operands)
            
            # Select best action according to policy
            with torch.no_grad():
                logits = self.model.forward([self.env.TOK_PAD] * self.env.max_length)
                masked_logits = torch.ones_like(logits) * float('-inf')
                for a in valid_actions:
                    masked_logits[a] = logits[a]
                action = torch.argmax(masked_logits).item() + 1
            
            # Process action
            tokens.append(action)
            if action == self.env.TOK_CONST:
                #statt zufälliger auswahl, erst platzhalter, der später optimiert wird
                const_value= 1.0
                self.constants.append(const_value)
            # Update required operands
            if action in self.env.binary_ops:
                required_operands += 1
            elif action in self.env.unary_ops:
                pass  # No change for unary
            else:
                required_operands -= 1
        
        # Check if expression is valid
        if required_operands != 0:
            return None
        
        # Set tokens in environment to generate expression string
        self.env.tokens = tokens
        self.env.required_operands = 0
        if self.env.tokens.count(self.env.TOK_CONST) > 0:
            self.env.optimize_constants()

        return self.env.get_expression_str()
    
    def optimize_best_expression(self):
        """Startet nach dem Training eine zusätzliche Konstanten-Optimierung."""
        if not self.best_expr:
            return

        # Parse the expression back into tokens
        try:
            self.env.reset()
            tokens = []
            constants = []

            # Simple parsing to extract constants from the expression
            expr = self.best_expr
            const_pattern = r'[-+]?\d*\.?\d+' # Pattern to match numbers

            # Find all constants in the expression
            const_matches = re.findall(const_pattern, expr)
            if const_matches:
                constants = [float(c) for c in const_matches]

            # Create a simplified version of the expression for optimization
            # This is a simplification - you might need more sophisticated parsing
            # based on your specific expression format
            for i, token in enumerate(self.env.tokens):
                if token == self.env.TOK_CONST:
                    if constants:
                        self.env.constants.append(constants.pop(0))
                    else:
                        self.env.constants.append(1.0) # Default if not enough constants

            # Run an extended optimization
            if self.env.optimize_constants():
                # Update best expression with optimized constants
                self.best_expr = self.env.get_expression_str()
                self.best_reward = self.env._calculate_reward()
                
                # Ausmultiplizieren des optimierten Ausdrucks
                simplified_expr = self.simplify_expression(self.best_expr)
                if simplified_expr:
                    self.best_expr = simplified_expr
                    
                print(f"Final optimization improved reward to: {self.best_reward:.3f}")

        except Exception as e:
            print(f"Error in final optimization: {e}")  


    def simplify_expression(self, expr_str):
                """Multipliziert einen Algebra-Ausdruck aus und vereinfacht ihn (sympy)."""
                if not expr_str:
                    return None
                
                try:
                    # Ersetze ^ durch ** für Python-Kompatibilität
                    expr_str = expr_str.replace('^', '**')
                    
                    # Definiere das Symbol x
                    x = sp.Symbol('x')
                    
                    # Konvertiere den String in einen sympy-Ausdruck
                    expr = sp.sympify(expr_str)
                    
                    # Ausmultiplizieren und vereinfachen
                    expanded_expr = sp.expand(expr)
                    
                    # Umwandlung zurück in String und ** wieder durch ^ ersetzen
                    expanded_str = str(expanded_expr).replace('**', '^')
                    
                    return expanded_str
                except Exception as e:
                    print(f"Fehler beim Vereinfachen des Ausdrucks: {e}")
                    return expr_str  # Gib den ursprünglichen Ausdruck zurück, wenn ein Fehler auftritt
                

# -------------------------------------------------
# MAIN FUNCTIONS
# -------------------------------------------------
def optimize_expression(target_fn, allowed_operators=None, episodes=EPISODES):
    """Startet den DSPTrainer und liefert besten Symbolic-Ausdruck + Reward."""
    trainer = DSPTrainer(target_fn, allowed_operators=allowed_operators, episodes=episodes)
    best_expr, best_reward = trainer.train()

    # Attempt to generate a better final expression
    final_expr = trainer.get_final_expression()
    if final_expr:
        # Verify if it's actually better
        trainer.env.reset()
        trainer.env.tokens = tokenize(final_expr)
        trainer.env.required_operands = 0
        final_reward = trainer.env._calculate_reward()
        
        if final_reward > best_reward:
            best_expr = final_expr
            best_reward = final_reward
            
            # Ausmultiplizieren des besten Ausdrucks am Ende
            simplified_expr = trainer.simplify_expression(best_expr)
            if simplified_expr:
                best_expr = simplified_expr

    return best_expr, best_reward

def plot_result(target_fn, expr_str, title="Function Approximation"):
    """Zeigt Ziel- und approximierte Funktion in einem Plot an."""
    # Generate x values for plotting
    if "sin" in str(target_fn) or "cos" in str(target_fn):
        x = torch.linspace(0, L, 200)
    else:
        x = torch.linspace(0, L, 200)
    
    # Evaluate target function
    if callable(target_fn):
        y_true = [target_fn(xi.item()) for xi in x]
    else:
        # If target_fn is a string
        expr = str(target_fn).replace('^', '**')
        y_true = [eval(expr, {"__builtins__": {}}, 
                       {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) 
                  for xi in x]
    
    # Evaluate approximated expression
    if expr_str:
        try:
            expr = expr_str.replace('^', '**')
            y_pred = [eval(expr, {"__builtins__": {}}, 
                            {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) 
                      for xi in x]
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            y_pred = [float('nan')] * len(x)
    else:
        y_pred = [float('nan')] * len(x)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_true, '-', color='black', label="Target")
    
    # Filter out nan/inf values
    x_np = x.numpy()
    valid_indices = [i for i, val in enumerate(y_pred) if not (math.isnan(val) or math.isinf(val))]
    if valid_indices:
        plt.plot(x_np[valid_indices], [y_pred[i] for i in valid_indices], '--', color='red', 
                 label=f"Approximation: {expr_str}")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_mse(target_fn, expr_str, x_range=(0, 5), n_points=200):
    """Berechnet den MSE zwischen Ziel- und Symbolic-Funktion auf gemeinsamem Gitter."""
    if not expr_str:
        return float('inf')
    
    try:
        # Verwende die gleichen x-Werte wie das PINN (0 bis L=5.0)
        x = torch.linspace(0, L, 200)  # Gleich wie x_dense im PINN Code
        
        # Evaluate target function (PINN Interpolator)
        if callable(target_fn):
            y_true = torch.tensor([float(target_fn(xi.item())) for xi in x])
        else:
            expr = str(target_fn).replace('^', '**')
            y_true = torch.tensor([eval(expr, {"__builtins__": {}},
                         {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()})
                     for xi in x])
        
        # Evaluate approximated expression
        expr = expr_str.replace('^', '**')
        y_pred = torch.tensor([eval(expr, {"__builtins__": {}},
                     {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()})
                 for xi in x])
        
        # Filter out invalid values
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y_true) | torch.isinf(y_true))
        if valid_mask.sum() == 0:
            return float('inf')
            
        mse = torch.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2).item()
        return mse
        
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return float('inf')
    

# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
if __name__ == "__main__":
    # Define target functions to approximate
    TARGET_FUNCTIONS = [
        ("PINN-Approximation", target_fn, ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("PINN-Approximation", target_fn, ['+', '-', '*', '^', 'sin', 'cos']),
        ("PINN-Approximation", target_fn, ['+', '-', '*', '^']),
        ("PINN-Approximation", target_fn, ['+', '-', '*']),
    ]
    
    results = {}
    plot_data = []  # Neue Liste zum Sammeln der Plot-Daten
    
    for i, (name, fn, operators) in enumerate(TARGET_FUNCTIONS):
        print(f"\n=== Optimizing f(x) = {name} (Run {i+1}/4) ===")
        best_expr, reward = optimize_expression(fn, allowed_operators=operators, episodes=EPISODES)
        
        # Calculate MSE
        mse = calculate_mse(fn, best_expr)
        
        print(f"→ Best expression: {best_expr}")
        print(f"→ Reward: {reward:.3f}")
        print(f"→ MSE: {mse:.6f}")
        
        results[f"{name}_Run_{i+1}"] = (best_expr, reward, mse)
        
        # Plot individual result immediately
        plot_result(fn, best_expr, f"{name} - Run {i+1} (MSE: {mse:.6f})")
        
        # Sammle Daten für zusammengefassten Plot
        plot_data.append((fn, best_expr, f"Run {i+1}", mse))
    
    # Print summary table with MSE
    print("\n=== Results Summary ===")
    print(f"{'Run':<15} | {'Approximation':<25} | {'Reward':<10} | {'MSE':<12}")
    print("-" * 70)
    for name, (expr, reward, mse) in results.items():
        print(f"{name:<15} | {expr if expr else 'Failed':<25} | {reward:.3f} | {mse:.6f}")
    
    # NEUER CODE: Zusammengefasster Plot aller 4 Ergebnisse
    plt.figure(figsize=(15, 10))
    
    for i, (target_fn, expr_str, run_name, mse) in enumerate(plot_data, 1):
        plt.subplot(2, 2, i)
        
        # Generate x values for plotting
        x = torch.linspace(0, L, 200)
        
        # Evaluate target function
        if callable(target_fn):
            y_true = [target_fn(xi.item()) for xi in x]
        else:
            expr = str(target_fn).replace('^', '**')
            y_true = [eval(expr, {"__builtins__": {}}, 
                           {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) 
                      for xi in x]
        
        # Evaluate approximated expression
        if expr_str:
            try:
                expr = expr_str.replace('^', '**')
                y_pred = [eval(expr, {"__builtins__": {}}, 
                                {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) 
                          for xi in x]
            except Exception as e:
                y_pred = [float('nan')] * len(x)
        else:
            y_pred = [float('nan')] * len(x)
        
        # Plot target and approximation
        plt.plot(x.numpy(), y_true, '-', color='black', label="Target", linewidth=2)
        
        # Filter out nan/inf values
        x_np = x.numpy()
        valid_indices = [j for j, val in enumerate(y_pred) if not (math.isnan(val) or math.isinf(val))]
        if valid_indices:
            plt.plot(x_np[valid_indices], [y_pred[j] for j in valid_indices], '--', color='red', 
                     label=f"Approximation", linewidth=2)
        
        plt.title(f"{run_name} (MSE: {mse:.6f})")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Symbolic Regression Results - All Runs", fontsize=16, y=1.02)
    plt.show()
    
    # Save results with MSE to file
    with open('symbolic_regression_results.txt', 'w') as f:
        f.write("Symbolic Regression Results\n")
        f.write("=========================\n\n")
        for name, (expr, reward, mse) in results.items():
            f.write(f"Run: {name}\n")
            f.write(f"Approximation: {expr if expr else 'Failed'}\n")
            f.write(f"Reward: {reward:.3f}\n")
            f.write(f"MSE: {mse:.6f}\n\n")
    
    print("\nResults saved to symbolic_regression_results.txt")
    print("All individual plots have been displayed during execution.")
    print("Combined plot with all 4 results has been shown at the end.")