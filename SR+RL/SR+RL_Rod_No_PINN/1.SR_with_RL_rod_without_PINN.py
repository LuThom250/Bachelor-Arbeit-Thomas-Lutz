import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
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

# -------------------------------------------------
# Problem­parameter DGL
# -------------------------------------------------
E  = 100.0   # Elastizitätsmodul
A  = 1.0     # Querschnittsfläche
q0 = 10.0    # Belastungskonstante
L  = 5.0     # Stablänge
bc_weight = 0.5  # Gewichtung der Randbedingungen im Verlust

# -------------------------------------------------
# Wichtige Hyperparameter RL & Symbolic Regression
# -------------------------------------------------
MAX_LENGTH    = 12      # Maximale Token-Länge eines Ausdrucks
X_RANGE       = (0, L)  # Betrachtung des Intervalls
N_POINTS      = 50      # Anzahl x-Stützstellen pro Episode
EPISODES      = 20000   # Gesamtzahl RL-Episoden
BATCH_SIZE    = 64      # Minibatch-Größe für Replay-Updates
LR            = 0.0001   # Lernrate des Policy-Netzes 
EMBED_SIZE    = 64      # Embedding dimension
HIDDEN_SIZE   = 128     # Hidden layer size
GAMMA         = 0.99    # Discount factor
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_END   = 0.1     # Final exploration rate
EPSILON_DECAY = 20000   # Exploration decay steps
ENTROPY_COEF  = 0.3     # Entropie-Bonus
MEMORY_SIZE   = 2000   # Kapazität des Replay Buffers
POLICY_UPDATE_FREQ = 5  # wie oft das Policy-Netz trainiert wird
TARGET_UPDATE_FREQ = 50 # wie oft die Target-Netz-Gewichte synchronisiert werden

# Sicherheits-/Stabilitätsparameter
SAFE_DIV_EPS  = 1e-6        # Division-Absicherung
EXP_CLAMP     =  (-20, 20)  # Overflow-Schwelle für exp-Aufruf
BASE_CLAMP    = (-100, 100) # Limit für Basen
POW_CLAMP     = (-5, 5)     # Begrenzung Exponent
GRAD_CLIP     = 1.0         # Gradient-Clipping
BONUS_SCALE   = 0.1         # Einfluss der Verbesserungs-Prämie

# Konstantenoptimierung: 0.0 entfernt, um konstante Null-Lösung zu vermeiden
CONST_DECIMALS = 2
CONST_FACTOR   = 10 ** CONST_DECIMALS
CONST_RANGE = [
    1.0, -1.0,
    0.5, -0.5,
    2.0, -2.0,
    3.14, -3.14,
    0.25, -0.25,
    5.0, -5.0,
    10.0, -10.0,
    math.pi, math.e
]
MAXITER_OPT    = 50

# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def print_table(data, headers=["Variable", "Wert"]):
    print(tabulate(data, headers=headers, tablefmt="grid"))

def print_highlighted(message, color=Fore.GREEN):
    print(color + message + Style.RESET_ALL)

def enhanced_progress_bar(iterable, desc="Verarbeitung läuft", unit=" Step"):
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
    def __init__(self, allowed_operators=None, x_range=X_RANGE, n_points=N_POINTS, max_length=MAX_LENGTH):
        self.x_range = x_range
        self.n_points = n_points
        self.max_length = max_length
        # Define tokens
        self.TOK_PAD   = 0
        self.TOK_X     = 1
        self.TOK_CONST = 2
        self.TOK_PLUS  = 3
        self.TOK_MINUS = 4
        self.TOK_MUL   = 5
        self.TOK_DIV   = 6
        self.TOK_POW   = 7
        self.TOK_SIN   = 8
        self.TOK_COS   = 9
        self.TOK_EXP   = 10
        # Allowed operators
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
        if self.allowed_operators:
            self.binary_ops = [self.op_mapping[op] for op in self.allowed_operators if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in self.allowed_operators if op in ['sin', 'cos', 'exp']]
        else:
            self.binary_ops = [self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW]
            self.unary_ops = [self.TOK_SIN, self.TOK_COS, self.TOK_EXP]
        self.terminals = [self.TOK_X, self.TOK_CONST]
        self.token_str = {
            self.TOK_X:     "x",
            self.TOK_CONST: "1",
            self.TOK_PLUS:  "+",
            self.TOK_MINUS: "-",
            self.TOK_MUL:   "*",
            self.TOK_DIV:   "/",
            self.TOK_POW:   "^",
            self.TOK_SIN:   "sin",
            self.TOK_COS:   "cos",
            self.TOK_EXP:   "exp",
        }
        self.vocab_size = len(self.token_str) + 1
        self.reset()
    
    def reset(self):
        self.tokens = []
        self.constants = []
        self.required_operands = 1
        self.steps = 0
        self.done = False
        X = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        self.x_values = X
        self.prev_normalized_loss = None
        return [self.TOK_PAD] * self.max_length
    
    def step(self, action_token):
        if self.done:
            raise RuntimeError("step() called after episode is done")
        self.steps += 1
        # Prüfe ungültige Nestung bei Unary
        if action_token in self.unary_ops:
            if self._is_inside_restricted_function():
                self.done = True
                return self._pad_obs(), -5.0, True, {'expr': None}
        # Exponent darf nicht Variable sein
        if len(self.tokens) > 0 and self.tokens[-1] == self.TOK_POW:
            if action_token == self.TOK_X:
                self.done = True
                return self._pad_obs(), -5.0, True, {'expr': None}
        # Konstantenaktion
        if action_token == self.TOK_CONST:
            const_value = random.choice(CONST_RANGE)
            self.constants.append(const_value)
        self.tokens.append(action_token)
        # Update Operanden-Zähler
        if action_token in self.binary_ops:
            self.required_operands += 1
        elif action_token in self.unary_ops:
            pass
        else:
            self.required_operands -= 1
        reward, info = 0.0, {}
        # Prüfe Ende der Episode
        if self.required_operands == 0 or self.steps >= self.max_length:
            self.done = True
            if self.required_operands == 0:
                if self._has_invalid_trig_expressions():
                    reward = -5.0
                    info['expr'] = None
                else:
                    expr_str = self.get_expression_str()
                    if expr_str:
                        # Verbot konstanter Lösungen ohne 'x'
                        if 'x' not in expr_str:
                            reward = -5.0
                            info['expr'] = None
                        else:
                            # Prüfe, ob Ausdruck funktional Null ist (z.B. x-x)
                            try:
                                expr_sym = sp.simplify(sp.sympify(expr_str.replace('^','**')))
                                # Wenn nach Vereinfachung eine Konstante (insb. 0) bleibt, verwerfen
                                if expr_sym.is_Number:
                                    reward = -5.0
                                    info['expr'] = None
                                else:
                                    reward = self._calculate_reward(expr_str)
                                    info['expr'] = expr_str
                            except Exception:
                                reward = -5.0
                                info['expr'] = None
                    else:
                        reward = -5.0
                        info['expr'] = None
            else:
                reward = -5.0
                info['expr'] = None
        obs = self._pad_obs()
        return obs, reward, self.done, info
    
    def _pad_obs(self):
        return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))
    
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
    
    def _calculate_reward(self, expr_str):
        try:
            expr_sym = sp.sympify(expr_str.replace('^', '**'))
            # Nach Vereinfachung prüfen, ob Nullfunktion
            expr_simp = sp.simplify(expr_sym)
            if expr_simp.is_Number:
                return -5.0
            x = sp.Symbol('x')
            u_sym = expr_sym
            # Ableitungen
            u_prime_sym = sp.diff(u_sym, x)
            u_double_sym = sp.diff(u_sym, x, 2)
            # Lambdify
            u_dd_func = sp.lambdify(x, u_double_sym, 'numpy')
            u_func = sp.lambdify(x, u_sym, 'numpy')
            u_d_func = sp.lambdify(x, u_prime_sym, 'numpy')
            # x-Werte: vermeide exakt 0, um Singularitäten zu umgehen
            eps = 1e-3
            x_np = np.linspace(self.x_range[0] + eps, self.x_range[1], self.n_points)
            # Evaluate unter Unterdrückung von Warnings
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_dd_vals = u_dd_func(x_np)
            # Prüfe komplex oder None
            if u_dd_vals is None or np.iscomplexobj(u_dd_vals):
                return -5.0
            residual = A * E * u_dd_vals + q0 * x_np
            if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
                return -5.0
            mse_res = float(np.mean(np.real(residual)**2))
            # Randbedingungen: u(0)=0, u'(L)=0
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u0 = u_func(0.0)
                uL_d = u_d_func(self.x_range[1])
            # Prüfe komplex oder invalid
            if np.iscomplex(u0) or np.iscomplex(uL_d):
                return -5.0
            try:
                u0_val = float(u0)
                uL_d_val = float(uL_d)
            except:
                return -5.0
            if math.isnan(u0_val) or math.isnan(uL_d_val) or math.isinf(u0_val) or math.isinf(uL_d_val):
                return -5.0
            bc_loss = u0_val**2 + uL_d_val**2
            total_loss = mse_res + bc_weight * bc_loss
            scale = np.mean((q0 * x_np)**2) + 1e-6
            normalized = total_loss / scale
            base_reward = math.exp(-normalized)
            if self.prev_normalized_loss is None:
                improvement = 0.0
            else:
                improvement = (self.prev_normalized_loss - normalized) / self.prev_normalized_loss if self.prev_normalized_loss > 0 else 0.0
            bonus = BONUS_SCALE * max(0, min(improvement, 1))
            self.prev_normalized_loss = normalized
            return base_reward + bonus
        except Exception:
            return -5.0
    
    def _evaluate_expression(self):
        raise NotImplementedError("_evaluate_expression not used in PDE mode")
    
    def get_expression_str(self):
        if self.required_operands != 0:
            return None
        const_idx = 0
        def build_expr(i):
            nonlocal const_idx
            if i >= len(self.tokens):
                return None, i
            tok = self.tokens[i]
            if tok in self.binary_ops:
                left_expr, next_i = build_expr(i+1)
                right_expr, next_i = build_expr(next_i)
                if left_expr is None or right_expr is None:
                    return None, next_i
                op = self.token_str[tok]
                if tok == self.TOK_POW:
                    return f"({left_expr})^({right_expr})", next_i
                else:
                    return f"({left_expr} {op} {right_expr})", next_i
            elif tok in self.unary_ops:
                arg_expr, next_i = build_expr(i+1)
                if arg_expr is None:
                    return None, next_i
                func = self.token_str[tok]
                return f"{func}({arg_expr})", next_i
            elif tok == self.TOK_X:
                return "x", i+1
            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return str(val), i+1
                else:
                    return None, i+1
            return None, i+1
        expr_str, _ = build_expr(0)
        return expr_str

# -------------------------------------------------
# POLICY NETWORK
# -------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size - 1)
        nn.init.zeros_(self.embed.weight[0])
        for param in self.lstm.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, seq):
        if isinstance(seq, torch.Tensor):
            t = seq
        else:
            t = torch.LongTensor(seq).unsqueeze(0)
        seq_list = t.tolist()[0] if t.dim() > 1 else t.tolist()
        seq_len = seq_list.index(0) if 0 in seq_list else len(seq_list)
        if seq_len == 0:
            h = torch.zeros((1, self.lstm.hidden_size))
        else:
            if t.dim() > 1:
                embedded = self.embed(t[:, :seq_len])
            else:
                embedded = self.embed(t[:seq_len].unsqueeze(0))
            output, (hidden, _) = self.lstm(embedded)
            h = output[:, -1, :]
        return self.fc(h).squeeze(0)
    
    def select_action(self, seq, valid_actions=None):
        with torch.no_grad():
            logits = self.forward(seq)
            if valid_actions is not None:
                mask = torch.zeros_like(logits)
                mask[valid_actions] = 1
                masked_logits = logits + (mask - 1) * 1e9
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
    def __init__(self, allowed_operators=None, episodes=EPISODES, lr=LR, entropy_coef=ENTROPY_COEF, batch_size=BATCH_SIZE):
        self.env = SymbolicRegressionEnv(allowed_operators=allowed_operators)
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
        valid_actions = []
        if required_operands > 0:
            valid_actions.extend(self.env.binary_ops)
            if not self.env._is_inside_restricted_function():
                valid_actions.extend(self.env.unary_ops)
        if required_operands > 0:
            if len(self.env.tokens) > 0 and self.env.tokens[-1] == self.env.TOK_POW:
                valid_actions.append(self.env.TOK_CONST)
            else:
                valid_actions.extend(self.env.terminals)
        return [a - 1 for a in valid_actions]
    
    def select_action(self, state, required_operands):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        valid_actions = self.get_valid_actions(required_operands)
        if random.random() < epsilon:
            action = random.choice(valid_actions) + 1
            return action, None, None
        else:
            return self.model.select_action(state, valid_actions)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)
        state_tensor = torch.LongTensor(state_batch)
        next_state_tensor = torch.LongTensor(next_state_batch)
        action_tensor = torch.LongTensor([a - 1 for a in action_batch])
        reward_tensor = torch.FloatTensor(reward_batch)
        done_tensor = torch.FloatTensor(done_batch)
        
        logits = self.model(state_tensor)
        if logits.dim() == 1:
            current_q_values = logits[action_tensor]
        else:
            current_q_values = logits.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_logits = self.target_model(next_state_tensor)
            if next_logits.dim() == 1:
                next_q_values = next_logits.max()
            else:
                next_q_values = next_logits.max(1)[0]
            target_q_values = reward_tensor + GAMMA * next_q_values * (1 - done_tensor)
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-GRAD_CLIP, GRAD_CLIP)
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train(self):
        for episode in enhanced_progress_bar(range(1, self.episodes + 1), desc="Training läuft", unit=" Ep."):
            state = self.env.reset()
            episode_reward = 0
            info = {}
            while not self.env.done:
                action, log_prob, entropy = self.select_action(state, self.env.required_operands)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
            expr = info.get('expr')
            if expr and episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_expr = expr
                try:
                    simplified = self.simplify_expression(self.best_expr)
                    if simplified:
                        self.best_expr = simplified
                except:
                    pass
                print_highlighted(f"Episode {episode}/{self.episodes} – Neuer bester Ausdruck: {self.best_expr} | Reward: {self.best_reward:.3f}", Fore.GREEN)
            self.history.append(episode_reward)
            if self.baseline is None:
                self.baseline = episode_reward
            else:
                self.baseline = 0.95 * self.baseline + 0.05 * episode_reward
            if episode % POLICY_UPDATE_FREQ == 0 and len(self.memory) >= self.batch_size:
                _ = self.optimize_model()
            if episode % TARGET_UPDATE_FREQ == 0:
                self.update_target_network()
            if episode % 100 == 0:
                recent = self.history[-100:]
                avg_reward = sum(recent) / len(recent) if recent else 0.0
                print(f"Episode {episode}/{self.episodes}, Avg reward: {avg_reward:.3f}, Best: {self.best_expr}")
        return self.best_expr, self.best_reward
    
    def get_final_expression(self):
        self.env.reset()
        tokens = []
        required_operands = 1
        while required_operands > 0 and len(tokens) < self.env.max_length:
            valid_actions = self.get_valid_actions(required_operands)
            with torch.no_grad():
                logits = self.model.forward([self.env.TOK_PAD] * self.env.max_length)
                masked_logits = torch.ones_like(logits) * float('-inf')
                for a in valid_actions:
                    masked_logits[a] = logits[a]
                action = torch.argmax(masked_logits).item() + 1
            tokens.append(action)
            if action == self.env.TOK_CONST:
                self.env.constants.append(1.0)
            if action in self.env.binary_ops:
                required_operands += 1
            elif action in self.env.unary_ops:
                pass
            else:
                required_operands -= 1
        if required_operands != 0:
            return None
        self.env.tokens = tokens
        self.env.required_operands = 0
        expr = self.env.get_expression_str()
        return expr
    
    def simplify_expression(self, expr_str):
        if not expr_str:
            return None
        try:
            expr_py = expr_str.replace('^', '**')
            x = sp.Symbol('x')
            expr = sp.sympify(expr_py)
            expanded = sp.expand(expr)
            return str(expanded).replace('**', '^')
        except Exception:
            return expr_str

# -------------------------------------------------
# MAIN-FUNKTION UND USAGE
# -------------------------------------------------
def analytic_solution(x_vals):
    # Analytische Lösung für E*A*u'' + q0*x = 0,
    # BC: u(0)=0, u'(L)=0
    # u(x) = -q0/(6*E*A)*x^3 + q0*L^2/(2*E*A)*x
    # return numpy array
    return -q0/(6*E*A) * x_vals**3 + q0*L**2/(2*E*A) * x_vals

def optimize_expression(allowed_operators=None, episodes=EPISODES):
    trainer = DSPTrainer(allowed_operators=allowed_operators)
    best_expr, best_reward = trainer.train()
    final_expr = trainer.get_final_expression()
    if final_expr:
        reward = trainer.env._calculate_reward(final_expr)
        if reward > best_reward:
            best_expr = final_expr
            best_reward = reward
            try:
                best_expr = trainer.simplify_expression(best_expr)
            except:
                pass
    return best_expr, best_reward

def plot_result(expr_str, title="PDE-Approximation"):
    x = torch.linspace(0, L, 200)
    x_np = x.numpy()
    # Netz-Vorhersage
    if expr_str:
        try:
            expr_py = expr_str.replace('^', '**')
            y_pred = np.array([eval(expr_py, {"__builtins__": {}}, 
                           {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': xi}) 
                      for xi in x_np])
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            y_pred = np.full_like(x_np, np.nan)
    else:
        y_pred = np.full_like(x_np, np.nan)
    # Analytische Lösung
    y_analytic = analytic_solution(x_np)
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(x_np, y_analytic, '-', label='Analytic Solution', linewidth=2)
    plt.plot(x_np, y_pred, '--', label=f"Approximation: {expr_str}", linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # Zusätzlich Fehlerplot
    if expr_str:
        error = y_pred - y_analytic
        plt.figure(figsize=(8,4))
        plt.plot(x_np, error, 'r-', label='Error (approx - analytic)')
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.title('Approximation Error vs Analytic')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    best_expr, best_reward = optimize_expression(allowed_operators=['+', '-', '*'], episodes=10000)
    print(f"Best expression: {best_expr}, Reward: {best_reward:.3f}")
    plot_result(best_expr, title=f"Approximation u(x), Expr: {best_expr}")
