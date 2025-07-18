# Alle Importe (PINN-bezogene entfernt, nur SR/RL-relevante)
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
# Hyperparameter für Symbolic Policy with Reinforcement Learning
# -------------------------------------------------
MAX_LENGTH    = 12      # Maximale Token-Länge eines Ausdrucks
X_RANGE       = (0, 5.0)  # Domain für x bei Auswertung der Ziel- und Kandidatfunktionen
N_POINTS      = 50      # Anzahl x-Stützstellen pro Episode
EPISODES      = 10000   # Gesamtzahl RL-Episoden zum Testen (z.B. 10000)
BATCH_SIZE    = 64      # Minibatch-Größe für Replay-Updates
LR            = 0.001   # Lernrate des Policy-Netzes 
EMBED_SIZE    = 64      # Embedding-Dimension
HIDDEN_SIZE   = 128     # Hidden-Layer-Größe
GAMMA         = 0.99    # Discount Factor
EPSILON_START = 1.0     # Initiale Exploration-Rate
EPSILON_END   = 0.2     # End-Exploration-Rate
EPSILON_DECAY = 50000   # Decay-Schritte für Epsilon
ENTROPY_COEF  = 0.3     # (derzeit ungenutzt) Entropie-Bonus
MEMORY_SIZE   = 10000   # Kapazität des Replay Buffers
POLICY_UPDATE_FREQ = 5  # wie oft (in Episoden) das Policy-Netz trainiert wird
TARGET_UPDATE_FREQ = 50 # wie oft Target-Netz synchronisiert wird

# ------- Sicherheits-/Stabilitätsparameter --------------------------
SAFE_DIV_EPS  = 1e-6        # Division-Absicherung
EXP_CLAMP     =  (-20, 20)  # Clamp für exp-Argumente
BASE_CLAMP    = (-100, 100) # Clamp Basis bei Potenz
POW_CLAMP     = (0, 6)     # Clamp Exponent bei Potenz
GRAD_CLIP     = 1.0         # Gradient Clipping
BONUS_SCALE   = 0.1         # Einfluss des Verbesserungsbonus im Reward

# ------- Konstantenoptimierung --------------------------------------
CONST_DECIMALS = 2                     
CONST_FACTOR   = 10 ** CONST_DECIMALS  
CONST_RANGE = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.14, -3.14, 
               0.25, -0.25, 5.0, -5.0, 10.0, -10.0, math.pi, math.e]
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

def enhanced_progress_bar(iterable, desc="Training läuft", unit=" Ep."):
    """Ersetzt die Standard-tqdm-Schleife durch eine erweiterte Fortschrittsanzeige."""
    return tqdm(iterable, desc=desc, unit=unit)

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
# ENVIRONMENT für Symbolic Regression
# -------------------------------------------------
class SymbolicRegressionEnv:
    def __init__(self, target_fn, allowed_operators=None, x_range=X_RANGE, n_points=N_POINTS, max_length=MAX_LENGTH):
        self.target_fn = target_fn
        self.x_range = x_range
        self.n_points = n_points
        self.max_length = max_length
        
        # Token-Definitions
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
            self.binary_ops = [self.op_mapping[op] for op in self.allowed_operators 
                               if op in ['+', '-', '*', '/', '^']]
            self.unary_ops  = [self.op_mapping[op] for op in self.allowed_operators 
                               if op in ['sin', 'cos', 'exp']]
        else:
            self.binary_ops = [self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW]
            self.unary_ops  = [self.TOK_SIN, self.TOK_COS, self.TOK_EXP]
        
        self.terminals = [self.TOK_X, self.TOK_CONST]
        self.token_str = {
            self.TOK_X:     "x",
            self.TOK_CONST: "1",  # Platzhalter, ersetzt beim Ausdrucksaufbau
            self.TOK_PLUS:  "+",
            self.TOK_MINUS: "-",
            self.TOK_MUL:   "*",
            self.TOK_DIV:   "/",
            self.TOK_POW:   "^",
            self.TOK_SIN:   "sin",
            self.TOK_COS:   "cos",
            self.TOK_EXP:   "exp",
        }
        self.vocab_size = len(self.token_str) + 1  # +1 für Padding
        self.reset()
    
    def reset(self):
        self.tokens = []
        self.constants = []
        self.required_operands = 1
        self.steps = 0
        self.done = False
        self._need_variable_in_trig = False
        # x-Werte für Reward-Berechnung
        X = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        self.x_values = X
        # Zielwerte berechnen
        try:
            if callable(self.target_fn):
                self.target_y = torch.tensor([float(self.target_fn(x.item())) for x in X])
            else:
                # Falls string-Ausdruck übergeben wurde
                self.target_y = self._eval_expr_string(self.target_fn, X)
        except Exception as e:
            print(f"Error evaluating target function: {e}")
            self.target_y = torch.zeros_like(X)
        self.prev_normalized_mse = 1.0
        return [self.TOK_PAD] * self.max_length
    
    def _eval_expr_string(self, expr_str, x_values):
        """Bewertet Ausdrucks-String elementweise (eval-safe)."""
        expr_str = expr_str.replace('^', '**')
        results = []
        for x in x_values:
            try:
                val = eval(expr_str, 
                           {"__builtins__": {}},
                           {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': x.item()})
                results.append(float(val))
            except:
                results.append(float('nan'))
        return torch.tensor(results)
    
    def step(self, action_token):
        if self.done:
            raise RuntimeError("step() called after episode is done")
        self.steps += 1
        
        # Ungültige Verschachtelungen verhindern
        if action_token in self.unary_ops:
            if self._is_inside_restricted_function():
                self.done = True
                return self._padded_obs(), 0.0, True, {'expr': None}
        # Für ^: nur Konstanten als Exponent zulassen
        if len(self.tokens) > 0 and self.tokens[-1] == self.TOK_POW:
            if action_token == self.TOK_X:
                self.done = True
                return self._padded_obs(), 0.0, True, {'expr': None}
        # Konstantenwahl
        if action_token == self.TOK_CONST:
            const_value = random.choice(CONST_RANGE)
            self.constants.append(const_value)
        # Token hinzufügen
        self.tokens.append(action_token)
        # Mögliche trig-invalidität merken
        if len(self.tokens) >= 2:
            if (self.tokens[-2] in [self.TOK_SIN, self.TOK_COS] and 
                self.tokens[-1] == self.TOK_CONST):
                self._need_variable_in_trig = True
        # required_operands aktualisieren
        if action_token in self.binary_ops:
            self.required_operands += 1
        elif action_token in self.unary_ops:
            pass
        else:
            self.required_operands -= 1
        
        reward, info = 0.0, {}
        # Prüfen, ob Ausdruck komplett oder Länge überschritten
        if self.required_operands == 0 or self.steps >= self.max_length:
            self.done = True
            if self.required_operands == 0:
                if self._has_invalid_trig_expressions():
                    reward = 0.0
                    info['expr'] = None
                else:
                    reward = self._calculate_reward()
                    info['expr'] = self.get_expression_str()
            else:
                reward = 0.0
                info['expr'] = None
        return self._padded_obs(), reward, self.done, info
    
    def _padded_obs(self):
        return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))
    
    def _is_inside_restricted_function(self):
        """Prüft geschachtelte Funktionsaufrufe, die verboten sind."""
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
        """Erkennt sin/cos ohne Variable."""
        stack = []
        trig_funcs = [self.TOK_SIN, self.TOK_COS]
        for token in self.tokens:
            if token in trig_funcs:
                stack.append((token, False))
            elif token == self.TOK_X and stack:
                func, _ = stack[-1]
                stack[-1] = (func, True)
            elif token in self.binary_ops:
                pass
            elif token in self.terminals and token != self.TOK_X:
                pass
        for func, has_var in stack:
            if func in trig_funcs and not has_var:
                return True
        return False
    
    def _calculate_reward(self):
        """Berechnet Reward aus MSE + Bonus."""
        try:
            y_pred = self._evaluate_expression()
        except Exception:
            return 0.0
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if y_pred.shape != self.target_y.shape:
            if y_pred.shape == torch.Size([]):
                y_pred = y_pred * torch.ones_like(self.target_y)
            else:
                return 0.0
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            return 0.0
        mse = float(torch.mean((y_pred - self.target_y) ** 2))
        variance = float(torch.var(self.target_y)) + 1e-6
        normalized_mse = mse / variance
        base_reward = math.exp(-normalized_mse)
        if self.prev_normalized_mse > 0:
            improvement_ratio = (self.prev_normalized_mse - normalized_mse) / self.prev_normalized_mse
        else:
            improvement_ratio = 0
        bonus = BONUS_SCALE * max(0, min(improvement_ratio, 1))
        self.prev_normalized_mse = normalized_mse
        reward = base_reward + bonus
        return max(reward, 0.0)
    
    def _evaluate_expression(self):
        """Parst und berechnet aktuellen Präfix-Token-Array."""
        const_idx = 0
        def eval_prefix(idx):
            nonlocal const_idx
            if idx >= len(self.tokens):
                raise ValueError("Unexpected end of expression")
            tok = self.tokens[idx]
            if tok in self.binary_ops:
                left, next_idx = eval_prefix(idx + 1)
                right, next_idx = eval_prefix(next_idx)
                if not isinstance(left, torch.Tensor): left = torch.tensor(left)
                if not isinstance(right, torch.Tensor): right = torch.tensor(right)
                if tok == self.TOK_DIV:
                    right_safe = torch.where(right == 0, torch.full_like(right, SAFE_DIV_EPS), right)
                    return left / right_safe, next_idx
                elif tok == self.TOK_PLUS:
                    return left + right, next_idx
                elif tok == self.TOK_MINUS:
                    return left - right, next_idx
                elif tok == self.TOK_POW:
                    left_safe = torch.clamp(left, BASE_CLAMP)
                    if not isinstance(right, torch.Tensor) or (isinstance(right, torch.Tensor) and right.numel() == 1):
                        right_safe = torch.clamp(right, POW_CLAMP)
                        power_safe = torch.where(
                            (left_safe < 0) & (right_safe != right_safe.round()),
                            torch.abs(left_safe) ** right_safe,
                            left_safe ** right_safe
                        )
                        return power_safe, next_idx
                    else:
                        return torch.ones_like(left), next_idx
                else:  # MUL
                    return left * right, next_idx
            elif tok in self.unary_ops:
                arg, next_idx = eval_prefix(idx + 1)
                if not isinstance(arg, torch.Tensor): arg = torch.tensor(arg)
                if tok in [self.TOK_SIN, self.TOK_COS]:
                    if arg.numel() > 1 and torch.all(arg == arg[0]):
                        return torch.zeros_like(self.x_values), next_idx
                if tok == self.TOK_SIN:
                    return torch.sin(arg), next_idx
                elif tok == self.TOK_COS:
                    return torch.cos(arg), next_idx
                else:  # EXP
                    safe_arg = torch.clamp(arg, EXP_CLAMP)
                    return torch.exp(safe_arg), next_idx
            elif tok == self.TOK_X:
                return self.x_values, idx + 1
            elif tok == self.TOK_CONST:
                if const_idx < len(self.constants):
                    val = self.constants[const_idx]
                    const_idx += 1
                    return torch.full_like(self.x_values, val), idx + 1
                else:
                    return torch.ones_like(self.x_values), idx + 1
            raise ValueError(f"Unknown token {tok}")
        result, _ = eval_prefix(0)
        return result
    
    def get_expression_str(self):
        """Gibt Token-Liste als lesbaren Infix-String zurück."""
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
            return "?", i + 1
        expr_str, _ = build_expr(0)
        return expr_str
    
    def optimize_constants(self):
        """Feintuning der Konstanten per SciPy minimize."""
        if not self.tokens or self.required_operands != 0:
            return False
        const_count = self.tokens.count(self.TOK_CONST)
        if const_count == 0:
            return False
        initial_constants = self.constants.copy()
        def error_function(const_values):
            rounded_values = [round(val * CONST_FACTOR) / CONST_FACTOR for val in const_values]
            self.constants = rounded_values
            try:
                y_pred = self._evaluate_expression()
                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    return 1e10
                mse = float(torch.mean((y_pred - self.target_y)**2))
                return mse
            except:
                return 1e10
            finally:
                self.constants = initial_constants.copy()
        best_result = None
        best_mse = float('inf')
        # Startpunkte
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
                    if result.success:
                        rounded_result = [round(val * CONST_FACTOR) / CONST_FACTOR for val in result.x]
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
                except:
                    continue
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
        self.fc = nn.Linear(hidden_size, vocab_size - 1)  # -1, da PAD nicht ausgewählt wird
        nn.init.zeros_(self.embed.weight[0])  # Zero für Padding
        for param in self.lstm.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, seq):
        if isinstance(seq, torch.Tensor):
            if seq.dim() > 2:
                batch_size = seq.size(0)
                reshaped = seq.reshape(batch_size, -1)
                embedded = self.embed(reshaped)
                output, (hidden, _) = self.lstm(embedded)
                return self.fc(hidden.squeeze(0))
            else:
                t = seq
        else:
            t = torch.LongTensor(seq).unsqueeze(0)
        seq_list = t.tolist()[0] if t.dim() > 1 else t.tolist()
        seq_len = seq_list.index(0) if 0 in seq_list else len(seq_list)
        if seq_len == 0:
            h = torch.zeros((1, self.lstm.hidden_size))
        else:
            embedded = self.embed(t[:, :seq_len] if t.dim() > 1 else t[:seq_len])
            output, (hidden, _) = self.lstm(embedded)
            h = output[:, -1, :] if output.dim() > 2 else output[:, -1]
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
# TRAINER mit Early-Stopping bei perfekter Approximation
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
            return random.choice(valid_actions) + 1, None, None
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
        for episode in enhanced_progress_bar(range(1, self.episodes + 1)):
            state = self.env.reset()
            episode_reward = 0.0
            while not self.env.done:
                action, log_prob, entropy = self.select_action(state, self.env.required_operands)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
            # Konstantenoptimierung für fertige Ausdrücke
            if self.env.done and info.get('expr') and self.env.tokens.count(self.env.TOK_CONST) > 0:
                best_reward_loc = episode_reward
                best_constants = self.env.constants.copy()
                for _ in range(3):
                    self.env.constants = [random.uniform(-5, 5) for _ in range(len(self.env.constants))]
                    if self.env.optimize_constants():
                        new_reward = self.env._calculate_reward()
                        if new_reward > best_reward_loc:
                            best_reward_loc = new_reward
                            best_constants = self.env.constants.copy()
                self.env.constants = best_constants
                episode_reward = best_reward_loc
                info['expr'] = self.env.get_expression_str()
            # Falls Ausdruck vollständig und besser als bisher, aktualisieren
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_expr = info.get('expr')
                if self.best_expr:
                    simplified = self.simplify_expression(self.best_expr)
                    if simplified:
                        self.best_expr = simplified
                    print_highlighted(f"Episode {episode}/{self.episodes} – Neuer bester Ausdruck: {self.best_expr} | Reward: {self.best_reward:.3f}")
                    # Early-Stopping: Wenn MSE exakt 0, beende Training vorzeitig
                    # Berechne MSE hier über helper-Funktion:
                    mse_now = calculate_mse(self.env.target_fn, self.best_expr)
                    if mse_now is not None and mse_now < 1e-8:
                        print_highlighted(f"Perfekte Approximation erreicht (MSE ~ 0). Stoppe Training nach Episode {episode}.")
                        break
            self.history.append(episode_reward)
            if self.baseline is None:
                self.baseline = episode_reward
            else:
                self.baseline = 0.95 * self.baseline + 0.05 * episode_reward
            if episode % POLICY_UPDATE_FREQ == 0 and len(self.memory) >= self.batch_size:
                _ = self.optimize_model()
            if episode % TARGET_UPDATE_FREQ == 0:
                self.update_target_network()
        # Optionale finale Konstantenoptimierung
        self.optimize_best_expression()
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
                # Platzhalter-Konstante
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
        if self.env.tokens.count(self.env.TOK_CONST) > 0:
            self.env.optimize_constants()
        return self.env.get_expression_str()
    
    def optimize_best_expression(self):
        if not self.best_expr:
            return
        try:
            # Einfacher Versuch: Konstanten im besten Ausdruck optimieren
            self.env.reset()
            # Token-Liste aus best_expr wird nicht direkt rekonstruiert, hier überspringen
            pass
        except Exception as e:
            print(f"Error in final optimization: {e}")
    
    def simplify_expression(self, expr_str):
        if not expr_str:
            return None
        try:
            expr_py = expr_str.replace('^', '**')
            x = sp.Symbol('x')
            expr = sp.sympify(expr_py)
            expanded = sp.expand(expr)
            return str(expanded).replace('**', '^')
        except Exception as e:
            print(f"Fehler beim Vereinfachen des Ausdrucks: {e}")
            return expr_str

# -------------------------------------------------
# MAIN FUNCTIONS
# -------------------------------------------------
def optimize_expression(target_fn, allowed_operators=None, episodes=EPISODES):
    """Startet den DSPTrainer und liefert besten Symbolic-Ausdruck + Reward."""
    trainer = DSPTrainer(target_fn, allowed_operators=allowed_operators, episodes=episodes)
    best_expr, best_reward = trainer.train()
    final_expr = trainer.get_final_expression()
    if final_expr:
        # Überprüfen, ob die finale Expression besser ist
        trainer.env.reset()
        trainer.env.tokens = tokenize(final_expr)
        trainer.env.required_operands = 0
        final_reward = trainer.env._calculate_reward()
        if final_reward > best_reward:
            best_expr = final_expr
            best_reward = final_reward
            simplified = trainer.simplify_expression(best_expr)
            if simplified:
                best_expr = simplified
    return best_expr, best_reward

def plot_result(target_fn, expr_str, title="Function Approximation"):
    """Zeigt Ziel- und approximierte Funktion in einem Plot an."""
    x = torch.linspace(X_RANGE[0], X_RANGE[1], 200)
    # Zielwerte
    if callable(target_fn):
        y_true = [target_fn(xi.item()) for xi in x]
    else:
        expr_py = str(target_fn).replace('^', '**')
        y_true = [eval(expr_py, {"__builtins__": {}}, 
                       {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': xi.item()})
                  for xi in x]
    # Approximationswerte
    if expr_str:
        try:
            expr_py = expr_str.replace('^', '**')
            y_pred = [eval(expr_py, {"__builtins__": {}},
                           {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': xi.item()})
                      for xi in x]
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            y_pred = [float('nan')] * len(x)
    else:
        y_pred = [float('nan')] * len(x)
    plt.figure(figsize=(8,5))
    plt.plot(x.numpy(), y_true, '-', color='black', label="Target", linewidth=2)
    x_np = x.numpy()
    valid_indices = [i for i, v in enumerate(y_pred) if not (math.isnan(v) or math.isinf(v))]
    if valid_indices:
        plt.plot(x_np[valid_indices], [y_pred[i] for i in valid_indices], '--', color='red', label=f"Approximation: {expr_str}", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_mse(target_fn, expr_str, x_range=X_RANGE, n_points=200):
    """Berechnet MSE zwischen Ziel- und Symbolic-Funktion."""
    if not expr_str:
        return float('inf')
    try:
        x = torch.linspace(x_range[0], x_range[1], n_points)
        if callable(target_fn):
            y_true = torch.tensor([float(target_fn(xi.item())) for xi in x])
        else:
            expr_py = str(target_fn).replace('^', '**')
            y_true = torch.tensor([eval(expr_py, {"__builtins__": {}},
                                {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': xi.item()})
                                   for xi in x])
        expr_py = expr_str.replace('^', '**')
        y_pred = torch.tensor([eval(expr_py, {"__builtins__": {}},
                             {**{k: getattr(math, k) for k in ['sin','cos','exp']}, 'x': xi.item()})
                         for xi in x])
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y_true) | torch.isinf(y_true))
        if valid_mask.sum() == 0:
            return float('inf')
        mse = torch.mean((y_true[valid_mask] - y_pred[valid_mask])**2).item()
        return mse
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return float('inf')

# -------------------------------------------------
# EXAMPLE USAGE für SR mit RL: Test mit einfachen und mittelschweren Funktionen
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
if __name__ == "__main__":
    # Definiere die bereits vorhandenen Ziel-Funktionen plus Polynome Grad 3–5
    TARGET_FUNCTIONS = [
        # Bestehende Ausdrücke beibehalten:
        ("sin(x)", lambda x: math.sin(x), ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("sin(x) + 0.5*x", lambda x: math.sin(x) + 0.5*x, ['+', '-', '*', '^', 'sin', 'cos']),
        ("exp(-x) * cos(x)", lambda x: math.exp(-x) * math.cos(x), ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        # Polynome Grad 2–5:
        ("x^2 + 2*x + 1", lambda x: x**2 + 2*x + 1, ['+', '-', '*', '^']),
        ("x^3", lambda x: x**3, ['+', '-', '*', '^']),
        ("x^3 - 3*x^2 + 2*x - 1", lambda x: x**3 - 3*x**2 + 2*x - 1, ['+', '-', '*', '^']),
        ("2*x^3 + x^2 - x + 5", lambda x: 2*x**3 + x**2 - x + 5, ['+', '-', '*', '^']),
        ("x^4", lambda x: x**4, ['+', '-', '*', '^']),
        ("3*x^4 + 2*x^3 - x^2 + x - 2", lambda x: 3*x**4 + 2*x**3 - x**2 + x - 2, ['+', '-', '*', '^']),
        ("x^5", lambda x: x**5, ['+', '-', '*', '^']),
        ("x^5 - 2*x^4 + x^3 - x + 2", lambda x: x**5 - 2*x**4 + x**3 - x + 2, ['+', '-', '*', '^']),
    ]

    results = {}
    for name, fn, ops in TARGET_FUNCTIONS:
        print(f"\n=== Optimizing target: {name} ===")
        best_expr, reward = optimize_expression(fn, allowed_operators=ops, episodes=EPISODES)
        mse = calculate_mse(fn, best_expr)
        print(f"→ Best expression for {name}: {best_expr}")
        print(f"→ Reward: {reward:.3f}, MSE: {mse:.6f}")
        plot_result(fn, best_expr, title=f"{name} Approximation (MSE={mse:.4f})")
        results[name] = (best_expr, reward, mse)

    # Zusammenfassung aller Läufe
    print("\n=== Zusammenfassung aller Ziele ===")
    for name, (expr, reward, mse) in results.items():
        print(f"{name:<40} | Expr: {expr:<30} | Reward: {reward:.3f} | MSE: {mse:.6f}")


