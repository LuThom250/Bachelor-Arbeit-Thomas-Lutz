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
import seaborn as sns # Hinzugefügt für die Heatmap

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
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

# Modell instanziieren
model = PINN(layers)

# -------------------------------------------------
# Physik-Residual der PDE
# -------------------------------------------------
def pde_residual(model, x):
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual = A * E * u_xx + q0 * x
    return residual

# -------------------------------------------------
# Verlustfunktion (Physik + Randbedingungen)
# -------------------------------------------------
def loss_function(model, x_f):
    f = pde_residual(model, x_f)
    loss_f = torch.mean(f**2)
    
    x0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = model(x0)
    loss_bc1 = u0**2
    
    xL = torch.tensor([[L]], requires_grad=True)
    uL = model(xL)
    uL_x = torch.autograd.grad(uL, xL, grad_outputs=torch.ones_like(uL), create_graph=True)[0]
    loss_bc2 = uL_x**2
    
    loss = loss_f + loss_bc1 + loss_bc2
    return loss

# -------------------------------------------------
# Training
# -------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
x_val = torch.linspace(0, L, 200).view(-1,1)
u_pred = model(x_val).detach().numpy()
plt.figure(figsize=(8,5))
plt.plot(x_val.numpy(), u_pred, label='PINN Output', linewidth=2)
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Approximierte Verschiebung $u(x)$')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------
# Dichte Auswertung + erneuter Plot
# -------------------------------------------------
x_dense = torch.linspace(0, L, 200).view(-1, 1)
u_dense = model(x_dense).detach().numpy()

# -------------------------------------------------
# Interpolations­funktion für beliebige x
# -------------------------------------------------
target_fn = interp1d(x_dense.numpy().flatten(), u_dense.flatten(), 
                     kind='cubic', fill_value="extrapolate")
print("Interpolierter Wert bei x=0.5:", target_fn(0.5))




# -------------------------------------------------
# Hyperparamter für Symbolic Policy with Reinforcement Learning
# -------------------------------------------------
MAX_LENGTH    = 12      
X_RANGE       = (0, L)  
N_POINTS      = 50      
EPISODES      = 10000   
BATCH_SIZE    = 64      
LR            = 0.001   
EMBED_SIZE    = 64      
HIDDEN_SIZE   = 128     
GAMMA         = 0.99    
EPSILON_START = 1.0     
EPSILON_END   = 0.2     
EPSILON_DECAY = 10000   
ENTROPY_COEF  = 0.3     
MEMORY_SIZE   = 10000   
POLICY_UPDATE_FREQ = 5  
TARGET_UPDATE_FREQ = 50 
PROB_LOG_FREQ = 250 # NEU: Frequenz für das Loggen der Aktionswahrscheinlichkeiten

# ------- Sicherheits-/Stabilitätsparameter --------------------------
SAFE_DIV_EPS  = 1e-6    
EXP_CLAMP     = (-20, 20)
BASE_CLAMP    = (-100, 100)
POW_CLAMP     = (-5, 5)   
GRAD_CLIP     = 1.0       
BONUS_SCALE   = 0.1       

# ------- Konstantenoptimierung --------------------------------------
CONST_DECIMALS = 2                     
CONST_FACTOR   = 10 ** CONST_DECIMALS  
CONST_RANGE = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.14, -3.14, 0.25, -0.25, 5.0, -5.0, 10.0, -10.0, math.pi, math.e]
MAXITER_OPT    = 100                    
                


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
        
        self.allowed_operators = allowed_operators
        self.op_mapping = {
            '+': self.TOK_PLUS, '-': self.TOK_MINUS, '*': self.TOK_MUL,
            '/': self.TOK_DIV, '^': self.TOK_POW, 'sin': self.TOK_SIN,
            'cos': self.TOK_COS, 'exp': self.TOK_EXP
        }
        
        if self.allowed_operators:
            self.binary_ops = [self.op_mapping[op] for op in self.allowed_operators if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in self.allowed_operators if op in ['sin', 'cos', 'exp']]
        else:
            self.binary_ops = [self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW]
            self.unary_ops = [self.TOK_SIN, self.TOK_COS, self.TOK_EXP]
        
        self.terminals = [self.TOK_X, self.TOK_CONST]
        
        self.token_str = {
            self.TOK_PAD: "PAD", self.TOK_X: "x", self.TOK_CONST: "C",
            self.TOK_PLUS: "+", self.TOK_MINUS: "-", self.TOK_MUL: "*",
            self.TOK_DIV: "/", self.TOK_POW: "^", self.TOK_SIN: "sin",
            self.TOK_COS: "cos", self.TOK_EXP: "exp",
        }
        self.vocab_size = len(self.token_str)
        self.reset()
        
    def reset(self):
        self.tokens = []
        self.constants = []
        self.required_operands = 1
        self.steps = 0
        self.done = False
        self._need_variable_in_trig = False
        X = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        self.x_values = X
        try:
            if callable(self.target_fn):
                self.target_y = torch.tensor([float(self.target_fn(x.item())) for x in X])
            else:
                self.target_y = self._eval_expr_string(self.target_fn, X)
        except Exception as e:
            print(f"Error evaluating target function: {e}")
            self.target_y = torch.zeros_like(X)
        self.prev_normalized_mse = 1.0
        return [self.TOK_PAD] * self.max_length

    
    def _eval_expr_string(self, expr_str, x_values):
        expr_str = expr_str.replace('^', '**')
        results = []
        for x in x_values:
            try:
                result = eval(expr_str, {"__builtins__": {}}, {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': x.item()})
                results.append(float(result))
            except:
                results.append(float('nan'))
        return torch.tensor(results)
    
    def step(self, action_token):
        if self.done:
            raise RuntimeError("step() called after episode is done")
        
        self.steps += 1
        
        if action_token in self.unary_ops:
            if self._is_inside_restricted_function():
                self.done = True
                return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens)), -5.0, True, {'expr': None}
        
        if len(self.tokens) > 0 and self.tokens[-1] == self.TOK_POW:
            if action_token == self.TOK_X:
                self.done = True
                return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens)), -5.0, True, {'expr': None}
        
        if action_token == self.TOK_CONST:
            const_value = random.choice(CONST_RANGE)
            self.constants.append(const_value)
        
        self.tokens.append(action_token)
        
        if len(self.tokens) >= 2:
            if (self.tokens[-2] in [self.TOK_SIN, self.TOK_COS] and self.tokens[-1] == self.TOK_CONST):
                self._need_variable_in_trig = True
        
        if action_token in self.binary_ops:
            self.required_operands += 1
        elif action_token in self.unary_ops:
            pass
        else:
            self.required_operands -= 1
        
        reward, info = 0.0, {}
        
        if self.required_operands == 0 or self.steps >= self.max_length:
            self.done = True
            if self.required_operands == 0:
                if self._has_invalid_trig_expressions():
                    reward = -5.0
                    info['expr'] = None
                else:
                    reward = self._calculate_reward()
                    info['expr'] = self.get_expression_str()
            else:
                reward = -5.0
                info['expr'] = None
        
        obs = self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))
        return obs, reward, self.done, info
    
    def _is_inside_restricted_function(self):
        stack = []
        for token in self.tokens:
            if token in self.unary_ops:
                stack.append(token)
            elif token in self.terminals:
                if stack:
                    if token == self.TOK_X and stack[-1] in [self.TOK_SIN, self.TOK_COS]:
                        stack.pop()
                    elif token == self.TOK_CONST and stack[-1] not in [self.TOK_SIN, self.TOK_COS]:
                        stack.pop()
                    else:
                        pass
                else:
                    pass
            elif token in self.binary_ops:
                pass
        return len(stack) > 0
    
    def _has_invalid_trig_expressions(self):
        stack = []
        trig_funcs = [self.TOK_SIN, self.TOK_COS]
        for token in self.tokens:
            if token in trig_funcs:
                stack.append((token, False))
            elif token == self.TOK_X and stack:
                func, _ = stack[-1]
                stack[-1] = (func, True)
        for func, has_var in stack:
            if func in trig_funcs and not has_var:
                return True
        return False   
    
    def _calculate_reward(self):
        try:
            y_pred = self._evaluate_expression()
        except Exception:
            return -5.0
        if not isinstance(y_pred, torch.Tensor): y_pred = torch.tensor(y_pred)
        if y_pred.shape != self.target_y.shape:
            if y_pred.shape == torch.Size([]): y_pred = y_pred * torch.ones_like(self.target_y)
            else: return -5.0
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any(): return -5.0

        mse = float(torch.mean((y_pred - self.target_y) ** 2))
        variance = float(torch.var(self.target_y)) + 1e-6
        normalized_mse = mse / variance
        base_reward = math.exp(-normalized_mse)

        if self.prev_normalized_mse > 0:
            improvement_ratio = (self.prev_normalized_mse - normalized_mse) / self.prev_normalized_mse
        else: improvement_ratio = 0
        bonus =  BONUS_SCALE * max(0, min(improvement_ratio, 1))
        self.prev_normalized_mse = normalized_mse
        reward = base_reward + bonus
        return reward

    
    def _evaluate_expression(self):
        const_idx = 0
        def eval_prefix(idx):
            nonlocal const_idx
            if idx >= len(self.tokens): raise ValueError("Unexpected end of expression")
            tok = self.tokens[idx]
            
            if tok in self.binary_ops:
                left, next_idx = eval_prefix(idx + 1)
                right, next_idx = eval_prefix(next_idx)
                if not isinstance(left, torch.Tensor): left = torch.tensor(left)
                if not isinstance(right, torch.Tensor): right = torch.tensor(right)
                
                if tok == self.TOK_DIV:
                    right_safe = torch.where(right == 0, torch.full_like(right, SAFE_DIV_EPS), right)
                    return left / right_safe, next_idx
                elif tok == self.TOK_PLUS: return left + right, next_idx
                elif tok == self.TOK_MINUS: return left - right, next_idx
                elif tok == self.TOK_POW:
                    left_safe = torch.clamp(left, *BASE_CLAMP)
                    if not isinstance(right, torch.Tensor) or (isinstance(right, torch.Tensor) and right.numel() == 1):
                        right_safe = torch.clamp(right, *POW_CLAMP)
                        power_safe = torch.where((left_safe < 0) & (right_safe != right_safe.round()), torch.abs(left_safe) ** right_safe, left_safe ** right_safe)
                        return power_safe, next_idx
                    else: return torch.ones_like(left), next_idx
                else: return left * right, next_idx
            
            elif tok in self.unary_ops:
                arg, next_idx = eval_prefix(idx + 1)
                if not isinstance(arg, torch.Tensor): arg = torch.tensor(arg)
                if tok in [self.TOK_SIN, self.TOK_COS]:
                    if arg.numel() > 1 and torch.all(arg == arg[0]): return torch.zeros_like(self.x_values), next_idx
                if tok == self.TOK_SIN: return torch.sin(arg), next_idx
                elif tok == self.TOK_COS: return torch.cos(arg), next_idx
                else: 
                    safe_arg = torch.clamp(arg, *EXP_CLAMP)
                    return torch.exp(safe_arg), next_idx
            
            elif tok == self.TOK_X: return self.x_values, idx + 1
            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return torch.full_like(self.x_values, val), idx + 1
                else: return torch.ones_like(self.x_values), idx + 1
            raise ValueError(f"Unknown token {tok}")
        result, _ = eval_prefix(0)
        return result
    
    def get_expression_str(self):
        if self.required_operands != 0: return None
        const_idx = 0
        
        def build_expr(i):
            nonlocal const_idx
            
            # --- START OF FIX ---
            # Add a boundary check to prevent crashes on malformed expressions.
            if i >= len(self.tokens):
                # This should not happen with valid expressions, but it adds stability.
                raise IndexError(f"Parser tried to read token at index {i}, but expression only has {len(self.tokens)} tokens.")
            # --- END OF FIX ---
                
            tok = self.tokens[i]
            
            if tok in self.binary_ops:
                op = self.token_str[tok]
                left_expr, next_i = build_expr(i + 1)
                right_expr, next_i = build_expr(next_i)
                if tok == self.TOK_POW: return f"({left_expr})^({right_expr})", next_i
                else: return f"({left_expr} {op} {right_expr})", next_i
            elif tok in self.unary_ops:
                func = self.token_str[tok]
                arg_expr, next_i = build_expr(i + 1)
                return f"{func}({arg_expr})", next_i
            elif tok == self.TOK_X: return "x", i + 1
            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return str(round(val, CONST_DECIMALS)), i + 1
                else: return "1", i + 1
            # This fallback should ideally not be reached with valid tokens.
            return "?", i + 1
            
        try:
            expr_str, _ = build_expr(0)
            return expr_str
        except IndexError as e:
            # Catch the specific error from our check and return None.
            # print(f"Debug: Failed to build expression string. Error: {e}")
            return None

    def optimize_constants(self):
        if not self.tokens or self.required_operands != 0: return False
        const_count = self.tokens.count(self.TOK_CONST)
        if const_count == 0: return False
        
        initial_constants = self.constants.copy()
        
        def error_function(const_values):
            rounded_values = [round(val * CONST_FACTOR) / CONST_FACTOR for val in const_values]
            self.constants = rounded_values
            try:
                y_pred = self._evaluate_expression()
                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any(): return 1e10
                mse = float(torch.mean((y_pred - self.target_y)**2))
                return mse
            except Exception: return 1e10
            finally: self.constants = initial_constants.copy()
        
        best_result = None
        best_mse = float('inf')
        starting_points = [[round(val * CONST_FACTOR) / CONST_FACTOR for val in initial_constants], [0.0] * const_count, [0.5] * const_count, [1.0] * const_count, [-0.5] * const_count, [-1.0] * const_count] + [[round(random.uniform(-5, 5) * CONST_FACTOR) / CONST_FACTOR for _ in range(const_count)] for _ in range(3)]
        methods = ['L-BFGS-B', 'BFGS', 'Nelder-Mead']
        
        for start_point in starting_points:
            for method in methods:
                try:
                    bounds = None if method in ['BFGS', 'Nelder-Mead'] else [(-10, 10)] * const_count
                    result = optimize.minimize(error_function, start_point, method=method, bounds=bounds, options={'maxiter': MAXITER_OPT})
                    if result.success:
                        rounded_result = [round(val * CONST_FACTOR) / CONST_FACTOR for val in result.x]
                        self.constants = rounded_result
                        try:
                            y_pred = self._evaluate_expression()
                            mse = float(torch.mean((y_pred - self.target_y)**2))
                            if mse < best_mse:
                                best_mse = mse
                                best_result = rounded_result
                        except: pass
                        finally: self.constants = initial_constants.copy()
                except Exception: continue
        
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
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # Ausgabe für alle Tokens außer PAD
        self.fc = nn.Linear(hidden_size, vocab_size - 1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, seq):
        # Sicherstellen, dass der Input ein LongTensor ist
        if not isinstance(seq, torch.Tensor):
            seq = torch.LongTensor(seq)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)

        embedded = self.embed(seq)
        output, (hidden, _) = self.lstm(embedded)
        # Nimm den letzten relevanten Output vor dem Padding
        # Dies ist eine Vereinfachung; für Batch-Verarbeitung mit unterschiedlichen Längen wäre eine Maske besser.
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)

    def select_action(self, seq, valid_actions=None):
        """Wählt nach Policy oder Zufall eine gültige nächste Aktion aus."""
        with torch.no_grad():
            logits = self.forward(seq)
            
            if valid_actions is not None:
                mask = torch.full_like(logits, -float('inf'))
                mask.scatter_(1, torch.tensor([valid_actions]), 0.0)
                masked_logits = logits + mask
                dist = torch.distributions.Categorical(logits=masked_logits)
            else:
                dist = torch.distributions.Categorical(logits=logits)
            
            action = dist.sample()
            # +1, da die Ausgabe des Netzes um 1 verschoben ist (kein PAD)
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
        
        # --- NEUE LISTEN FÜR ANALYSE ---
        self.history = [] # Reward-Verlauf
        self.top_solutions = [] # Für die Top 3 Lösungen (reward, expression)
        self.seen_expressions = set() # Um Duplikate in top_solutions zu vermeiden
        self.entropy_history = [] # Policy-Entropie pro Episode
        self.action_probs_history = [] # Aktionswahrscheinlichkeiten zu bestimmten Zeitpunkten
        self.prob_log_epochs = [] # Episoden, in denen geloggt wurde

    def get_valid_actions(self, required_operands):
        valid_actions = []
        # Binäre Operatoren sind erlaubt, wenn wir einen Operanden haben und noch Platz ist
        if required_operands >= 1 and len(self.env.tokens) < self.env.max_length -1:
            valid_actions.extend(self.env.binary_ops)
        # Unäre Operatoren ebenfalls
        if required_operands >= 1 and len(self.env.tokens) < self.env.max_length -1:
            if not self.env._is_inside_restricted_function():
                valid_actions.extend(self.env.unary_ops)
        
        # Terminals (x, const) sind erlaubt, wenn wir einen Operanden benötigen
        if required_operands > 0:
            if len(self.env.tokens) > 0 and self.env.tokens[-1] == self.env.TOK_POW:
                valid_actions.append(self.env.TOK_CONST)
            else:
                valid_actions.extend(self.env.terminals)
        
        # Konvertiere zu Indizes für das Policy-Netz (Token-ID - 1, da PAD nicht gewählt wird)
        return [a - 1 for a in valid_actions if a > 0]

    def select_action(self, state, required_operands):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        valid_actions = self.get_valid_actions(required_operands)
        if not valid_actions: # Fallback, wenn keine gültige Aktion möglich ist
             return None, None, None

        if random.random() < epsilon:
            action = random.choice(valid_actions) + 1
            return action, None, None # Kein log_prob/entropy bei zufälliger Aktion
        else:
            return self.model.select_action(state, valid_actions)

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return None
        
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)
        
        state_tensor = torch.LongTensor(state_batch)
        next_state_tensor = torch.LongTensor(next_state_batch)
        action_tensor = torch.LongTensor([a - 1 for a in action_batch]).unsqueeze(1)
        reward_tensor = torch.FloatTensor(reward_batch)
        done_tensor = torch.FloatTensor(done_batch)
        
        current_q_values = self.model(state_tensor).gather(1, action_tensor)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_tensor).max(1)[0].detach()
            target_q_values = reward_tensor + GAMMA * next_q_values * (1 - done_tensor)
            
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-GRAD_CLIP, GRAD_CLIP)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _update_top_solutions(self, reward, expr):
        if expr is None or expr in self.seen_expressions:
            return

        self.seen_expressions.add(expr)
        
        if len(self.top_solutions) < 3:
            self.top_solutions.append((reward, expr))
            self.top_solutions.sort(key=lambda x: x[0], reverse=True)
        elif reward > self.top_solutions[-1][0]:
            self.top_solutions.pop()
            self.top_solutions.append((reward, expr))
            self.top_solutions.sort(key=lambda x: x[0], reverse=True)

    def train(self):
        for episode in enhanced_progress_bar(range(1, self.episodes + 1), desc="Training läuft", unit=" Ep."):
            state = self.env.reset()
            episode_reward = 0
            episode_entropies = []

            while not self.env.done:
                action, log_prob, entropy = self.select_action(state, self.env.required_operands)
                if action is None: break # Episode abbrechen, wenn keine Aktion möglich
                
                if entropy is not None:
                    episode_entropies.append(entropy.item())

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
            
            # --- Analyse-Daten sammeln ---
            self.history.append(episode_reward)
            if episode_entropies:
                self.entropy_history.append(np.mean(episode_entropies))
            else:
                self.entropy_history.append(0) # Falls nur zufällige Aktionen

            # Konstanten optimieren und Top-Lösungen aktualisieren
            if self.env.done and info.get('expr'):
                if self.env.tokens.count(self.env.TOK_CONST) > 0:
                    if self.env.optimize_constants():
                         # Nach Optimierung Reward und Ausdruck neu berechnen
                        episode_reward = self.env._calculate_reward()
                        info['expr'] = self.env.get_expression_str()
                
                # Top-Lösungen aktualisieren
                simplified_expr = self.simplify_expression(info.get('expr'))
                self._update_top_solutions(episode_reward, simplified_expr)

            if episode % POLICY_UPDATE_FREQ == 0: self.optimize_model()
            if episode % TARGET_UPDATE_FREQ  == 0: self.update_target_network()

            # --- NEU: Periodisches Loggen der Aktionswahrscheinlichkeiten ---
            if episode % PROB_LOG_FREQ == 0:
                self.prob_log_epochs.append(episode)
                with torch.no_grad():
                    initial_state = torch.LongTensor([self.env.TOK_PAD] * self.env.max_length)
                    logits = self.model(initial_state)
                    # Softmax anwenden, um Wahrscheinlichkeiten zu erhalten
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    self.action_probs_history.append(probs)

            if episode % 1000 == 0:
                avg_reward = np.mean(self.history[-100:])
                best_expr_so_far = self.top_solutions[0][1] if self.top_solutions else "None"
                print(f"\nEp {episode}/{self.episodes}, Avg reward: {avg_reward:.3f}, Best: {best_expr_so_far}")

        return self.top_solutions

    def simplify_expression(self, expr_str):
        if not expr_str: return None
        try:
            expr_str_py = expr_str.replace('^', '**')
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str_py, locals={'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
            expanded_expr = sp.expand(expr)
            return str(expanded_expr).replace('**', '^')
        except Exception: return expr_str
                
# -------------------------------------------------
# NEUE PLOTTING FUNKTIONEN
# -------------------------------------------------
def moving_average(data, window_size):
    """Berechnet den gleitenden Durchschnitt."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_reward_history(history, run_name):
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

def plot_epsilon_decay():
    epsilons = [EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-i / EPSILON_DECAY) for i in range(EPISODES)]
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons)
    plt.title('Epsilon-Decay (Explorationsrate)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.show()

def plot_entropy_history(entropy_history, run_name):
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
    if not action_probs: return
    action_probs_np = np.array(action_probs)
    # Token-Labels für die x-Achse (ohne PAD)
    action_labels = [env.token_str[i] for i in range(1, env.vocab_size)]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(action_probs_np, cmap='viridis', xticklabels=action_labels, yticklabels=epochs)
    plt.title(f'Heatmap der Aktionswahrscheinlichkeiten - {run_name}')
    plt.xlabel('Aktion (Token)')
    plt.ylabel('Episode')
    plt.show()

def plot_token_type_probs(action_probs, epochs, env, run_name):
    if not action_probs: return
    
    terminals_idx = [t - 1 for t in env.terminals]
    binary_ops_idx = [t - 1 for t in env.binary_ops]
    unary_ops_idx = [t - 1 for t in env.unary_ops]

    prob_hist = {'Terminal': [], 'Binary Op': [], 'Unary Op': []}
    
    for probs in action_probs:
        prob_hist['Terminal'].append(np.sum(probs[terminals_idx]))
        # Überprüfen ob Operatoren überhaupt erlaubt sind
        if binary_ops_idx:
            prob_hist['Binary Op'].append(np.sum(probs[binary_ops_idx]))
        if unary_ops_idx:
            prob_hist['Unary Op'].append(np.sum(probs[unary_ops_idx]))

    plt.figure(figsize=(12, 6))
    for name, values in prob_hist.items():
        if values: # Nur plotten, wenn der Typ im Lauf vorhanden war
            plt.plot(epochs, values, label=name, marker='o', linestyle='--')
    
    plt.title(f'Durchschnittliche Wahrscheinlichkeit nach Token-Typ - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Summierte Wahrscheinlichkeit')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()


# -------------------------------------------------
# MAIN FUNCTIONS
# -------------------------------------------------
def optimize_expression(target_fn, allowed_operators=None, episodes=EPISODES, run_name=""):
    """Startet den DSPTrainer und liefert die besten Symbolic-Ausdrücke."""
    print_highlighted(f"\n=== Starte Lauf: {run_name} ===", Fore.CYAN)
    trainer = DSPTrainer(target_fn, allowed_operators=allowed_operators, episodes=episodes)
    top_solutions = trainer.train()

    # --- ANZEIGE DER NEUEN PLOTS ---
    print_highlighted(f"\n--- Analysediagramme für Lauf: {run_name} ---", Fore.YELLOW)
    plot_reward_history(trainer.history, run_name)
    plot_entropy_history(trainer.entropy_history, run_name)
    plot_action_prob_heatmap(trainer.action_probs_history, trainer.prob_log_epochs, trainer.env, run_name)
    plot_token_type_probs(trainer.action_probs_history, trainer.prob_log_epochs, trainer.env, run_name)
    
    return top_solutions

def plot_result(target_fn, expr_str, title="Function Approximation"):
    """Zeigt Ziel- und approximierte Funktion in einem Plot an."""
    x = torch.linspace(0, L, 200)
    
    if callable(target_fn):
        y_true = [target_fn(xi.item()) for xi in x]
    else:
        expr = str(target_fn).replace('^', '**')
        y_true = [eval(expr, {"__builtins__": {}}, {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) for xi in x]
    
    if expr_str:
        try:
            expr = expr_str.replace('^', '**')
            y_pred = [eval(expr, {"__builtins__": {}}, {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) for xi in x]
        except Exception:
            y_pred = [float('nan')] * len(x)
    else: y_pred = [float('nan')] * len(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_true, '-', color='black', label="Target")
    
    x_np = x.numpy()
    valid_indices = [i for i, val in enumerate(y_pred) if not (math.isnan(val) or math.isinf(val))]
    if valid_indices:
        plt.plot(x_np[valid_indices], [y_pred[i] for i in valid_indices], '--', color='red', label=f"Approximation: {expr_str}")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_mse(target_fn, expr_str):
    if not expr_str: return float('inf')
    try:
        x = torch.linspace(0, L, 200)
        if callable(target_fn):
            y_true = torch.tensor([float(target_fn(xi.item())) for xi in x])
        else:
            expr = str(target_fn).replace('^', '**')
            y_true = torch.tensor([eval(expr, {"__builtins__": {}}, {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) for xi in x])
        
        expr = expr_str.replace('^', '**')
        y_pred = torch.tensor([eval(expr, {"__builtins__": {}}, {**{k: getattr(math, k) for k in ['sin', 'cos', 'exp']}, 'x': xi.item()}) for xi in x])
        
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y_true) | torch.isinf(y_true))
        if valid_mask.sum() == 0: return float('inf')
        mse = torch.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2).item()
        return mse
    except Exception: return float('inf')

# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
if __name__ == "__main__":
    TARGET_FUNCTIONS = [
        ("Run 1: Alle Operatoren", target_fn, ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("Run 2: Ohne exp", target_fn, ['+', '-', '*', '^', 'sin', 'cos']),
        ("Run 3: Nur Polynome", target_fn, ['+', '-', '*', '^']),
        ("Run 4: Nur lineare Terme", target_fn, ['+', '-', '*']),
    ]
    
    results = {}
    
    # Globaler Plot für Epsilon-Decay (ist für alle Läufe gleich)
    plot_epsilon_decay()

    for name, fn, operators in TARGET_FUNCTIONS:
        top_solutions = optimize_expression(fn, allowed_operators=operators, episodes=EPISODES, run_name=name)
        
        # Berechne MSE für die Top-Lösungen und speichere sie
        results[name] = []
        if top_solutions:
            for reward, expr in top_solutions:
                mse = calculate_mse(fn, expr)
                results[name].append({'expr': expr, 'reward': reward, 'mse': mse})
                # Plotte nur das beste Ergebnis des Laufs
            best_expr = top_solutions[0][1]
            best_mse = results[name][0]['mse']
            plot_result(fn, best_expr, f"Beste Approximation - {name} (MSE: {best_mse:.6f})")
        else:
            print(f"Lauf {name} hat keine gültige Lösung gefunden.")

    # Print summary table with MSE
    print("\n\n" + "="*80)
    print_highlighted("=== ZUSAMMENFASSUNG ALLER LÄUFE ===", Fore.MAGENTA)
    print("="*80)
    for name, solutions in results.items():
        print(f"\n--- {name} ---")
        if solutions:
            # Erstelle Daten für die tabulate-Bibliothek
            table_data = [[f"#{i+1}", s['expr'], f"{s['reward']:.4f}", f"{s['mse']:.6e}"] for i, s in enumerate(solutions)]
            print(tabulate(table_data, headers=["Rang", "Ausdruck", "Reward", "MSE"], tablefmt="grid"))
        else:
            print("Keine Lösung gefunden.")
    print("\n" + "="*80)