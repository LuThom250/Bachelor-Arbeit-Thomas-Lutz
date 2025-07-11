# Alle Importe
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
EPISODES      = 20000   # Gesamtzahl RL-Episoden zum Testen
X_RANGE       = (0.1, 5.0)  # Domain für x (0.1 statt 0, um Singularitäten bei 1/x zu vermeiden)
N_POINTS      = 50      # Anzahl x-Stützstellen pro Episode
BATCH_SIZE    = 64      # Minibatch-Größe für Replay-Updates
LR            = 0.001   # Lernrate des Policy-Netzes 
EMBED_SIZE    = 64      # Embedding-Dimension
HIDDEN_SIZE   = 128     # Hidden-Layer-Größe
GAMMA         = 0.99    # Discount Factor
EPSILON_START = 1.0     # Initiale Exploration-Rate
EPSILON_END   = 0.1     # End-Exploration-Rate
EPSILON_DECAY = 20000   # Decay-Schritte für Epsilon
MEMORY_SIZE   = 20000   # Kapazität des Replay Buffers
POLICY_UPDATE_FREQ = 5  # wie oft (in Episoden) das Policy-Netz trainiert wird
TARGET_UPDATE_FREQ = 50 # wie oft Target-Netz synchronisiert wird

# ------- Sicherheits-/Stabilitätsparameter --------------------------
SAFE_DIV_EPS  = 1e-8        # Division-Absicherung
EXP_CLAMP     = (-15, 15)   # Clamp für exp-Argumente
POW_CLAMP     = (0, 6)     # Clamp Exponent bei Potenz
GRAD_CLIP     = 1.0         # Gradient Clipping
PENALTY       = -1e9        # Hohe Strafe für ungültige/instabile Ausdrücke

# ------- Konstantenoptimierung --------------------------------------
CONST_DECIMALS = 3                   
CONST_FACTOR   = 10 ** CONST_DECIMALS  
CONST_RANGE = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, np.pi, np.e]
MAXITER_OPT    = 50                   

# ------- NEU: Hyperparameter für Curriculum Learning --------------------
MAX_LENGTH_LIMIT = 15      # Absolute Obergrenze für die Token-Länge
CURRICULUM_START_LENGTH = 5 # Starte mit dieser maximalen Länge
CURRICULUM_UPDATE_FREQ = 1000 # Alle X Episoden prüfen, ob Komplexität erhöht werden soll
CURRICULUM_STAGNATION_THRESHOLD = 0.01
CORRICLUM_STEP = 2  # Wenn Reward sich um weniger als 1% verbessert, erhöhe Komplexität


# ------- Berechnung Reward --------------------
beta = 1.0  # Skaliert den Einfluss des MSE
alpha = 0.01 # Strafe für Komplexität (etwas höher angesetzt)
# -------------------------------------------------
# Hilfsfunktionen (unverändert)
# -------------------------------------------------
def print_table(data, headers=["Variable", "Wert"]):
    """Gibt die Daten als übersichtliche Tabelle aus."""
    print(tabulate(data, headers=headers, tablefmt="grid"))

def print_highlighted(message, color=Fore.GREEN):
    """Gibt eine Nachricht farbig hervorgehoben aus."""
    print(color + message + Style.RESET_ALL)

def enhanced_progress_bar(iterable, desc="Training läuft", unit=" Ep."):
    """Ersetzt die Standard-tqdm-Schleife durch eine erweiterte Fortschrittsanzeige."""
    return tqdm(iterable, desc=desc, unit=unit, ncols=100)

# -------------------------------------------------
# TOKENIZATION (unverändert)
# -------------------------------------------------
TOKEN_PATTERN = r"\s*(sin|cos|exp|[()+\-*/^]|[0-9]+(?:\.[0-9]+)?|x)"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

def tokenize(expr):
    """Zerlegt einen Formel-String in Einzel-Tokens."""
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
    # MODIFIZIERT: max_length wird jetzt dynamisch übergeben
    def __init__(self, target_fn, allowed_operators=None, x_range=X_RANGE, n_points=N_POINTS, max_length=MAX_LENGTH_LIMIT):
        self.target_fn = target_fn
        self.x_range = x_range
        self.n_points = n_points
        self.max_length = max_length # MODIFIZIERT
        self.TOK_PAD, self.TOK_X, self.TOK_CONST, self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW, self.TOK_SIN, self.TOK_COS, self.TOK_EXP = range(11)
        self.op_mapping = {'+': self.TOK_PLUS, '-': self.TOK_MINUS, '*': self.TOK_MUL, '/': self.TOK_DIV, '^': self.TOK_POW, 'sin': self.TOK_SIN, 'cos': self.TOK_COS, 'exp': self.TOK_EXP}
        
        if allowed_operators:
            self.binary_ops = [self.op_mapping[op] for op in allowed_operators if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in allowed_operators if op in ['sin', 'cos', 'exp']]
        else:
            self.binary_ops = [self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW]
            self.unary_ops = [self.TOK_SIN, self.TOK_COS, self.TOK_EXP]
            
        self.terminals = [self.TOK_X, self.TOK_CONST]
        self.token_str = {v: k for k, v in {**self.op_mapping, "x": self.TOK_X, "1": self.TOK_CONST}.items()}
        self.vocab_size = len(self.token_str) + 1
        self.reset()
    
    def reset(self):
        self.tokens, self.constants = [], []
        self.required_operands, self.steps, self.done = 1, 0, False
        self.x_values = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        try:
            self.target_y = torch.tensor([float(self.target_fn(x.item())) for x in self.x_values])
        except Exception as e:
            print(f"Error evaluating target function: {e}")
            self.target_y = torch.zeros_like(self.x_values)
        return self._padded_obs()

    def step(self, action_token):
        if self.done: raise RuntimeError("step() called after episode is done")
        self.steps += 1
        
        if action_token == self.TOK_CONST: self.constants.append(random.choice(CONST_RANGE))
        self.tokens.append(action_token)

        if action_token in self.binary_ops: self.required_operands += 1
        elif action_token not in self.unary_ops: self.required_operands -= 1
        
        reward, info = 0.0, {}
        is_complete = self.required_operands == 0
        is_too_long = self.steps >= self.max_length # MODIFIZIERT: Prüft gegen die *aktuelle* maximale Länge
        
        if is_complete or is_too_long:
            self.done = True
            if is_complete:
                reward = self._calculate_reward()
                info['expr'] = self.get_expression_str()
            else:
                reward = PENALTY
                info['expr'] = None
                
        return self._padded_obs(), reward, self.done, info
    
    def _padded_obs(self):
        # MODIFIZIERT: Padding passt sich der dynamischen max_length an
        return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))

    def _calculate_reward(self):
        # NEU: Verbesserte, normierte Belohnungsfunktion für stabilieres Lernen
        try:
            y_pred = self._evaluate_expression()
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                return PENALTY
            mse = float(torch.mean((y_pred - self.target_y) ** 2))
            
            
            reward = math.exp(-beta * mse) - alpha * len(self.tokens)
            return reward
        except (ValueError, RuntimeError):
            return PENALTY

    def _evaluate_expression(self):
        const_idx = 0
        memo = {} # Hinzufügen von Memoization, um die Performance bei Rekursion zu verbessern
        def eval_prefix(idx):
            nonlocal const_idx
            if idx in memo: return memo[idx] # Memoization Check
            
            if idx >= len(self.tokens):
                raise ValueError("Unexpected end of expression during evaluation")
            
            tok = self.tokens[idx]
            
            if tok in self.binary_ops:
                # KORREKTE INDEX-BEHANDLUNG
                left, next_idx_after_left = eval_prefix(idx + 1)
                right, next_idx_after_right = eval_prefix(next_idx_after_left)
                
                if tok == self.TOK_DIV: 
                    result = left / (right + SAFE_DIV_EPS)
                elif tok == self.TOK_PLUS: 
                    result = left + right
                elif tok == self.TOK_MINUS: 
                    result = left - right
                elif tok == self.TOK_MUL: 
                    result = left * right
                elif tok == self.TOK_POW:
                    right_clamped = torch.clamp(right, POW_CLAMP[0], POW_CLAMP[1])
                    # Stabilitäts-Fix für negative Basen
                    base_safe = torch.abs(left) + SAFE_DIV_EPS 
                    result = torch.pow(base_safe, right_clamped)
                
                memo[idx] = (result, next_idx_after_right)
                return result, next_idx_after_right

            elif tok in self.unary_ops:
                arg, next_idx = eval_prefix(idx + 1)
                if tok == self.TOK_SIN: result = torch.sin(arg)
                elif tok == self.TOK_COS: result = torch.cos(arg)
                elif tok == self.TOK_EXP: result = torch.exp(torch.clamp(arg, EXP_CLAMP[0], EXP_CLAMP[1]))
                
                memo[idx] = (result, next_idx)
                return result, next_idx

            elif tok == self.TOK_X:
                memo[idx] = (self.x_values, idx + 1)
                return self.x_values, idx + 1
                
            elif tok == self.TOK_CONST:
                val = self.constants[const_idx] if const_idx < len(self.constants) else 1.0
                const_idx += 1
                result = torch.full_like(self.x_values, val)
                memo[idx] = (result, idx + 1)
                return result, idx + 1
                
            raise ValueError(f"Unknown token {tok}")

        result, _ = eval_prefix(0)
        return result
    
    def get_expression_str(self):
        # Unverändert
        if self.required_operands != 0: return None
        const_idx = 0
        def build_expr(i):
            nonlocal const_idx
            tok = self.tokens[i]
            if tok in self.binary_ops:
                op = self.token_str[tok]
                left, n1 = build_expr(i + 1)
                right, n2 = build_expr(n1)
                return f"({left} {op} {right})", n2
            elif tok in self.unary_ops:
                func, (arg, n) = self.token_str[tok], build_expr(i + 1)
                return f"{func}({arg})", n
            elif tok == self.TOK_X: return "x", i + 1
            elif tok == self.TOK_CONST:
                val = self.constants[const_idx] if const_idx < len(self.constants) else 1.0
                const_idx += 1
                return f"{val:.{CONST_DECIMALS}f}", i + 1
        expr_str, _ = build_expr(0)
        return expr_str
    
    def optimize_constants(self):
        # Unverändert
        const_count = self.tokens.count(self.TOK_CONST)
        if const_count == 0: return False, 0.0
        
        initial_constants = self.constants[:const_count]

        def error_function(const_values):
            self.constants = list(const_values)
            try:
                y_pred = self._evaluate_expression()
                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any(): return -PENALTY
                return float(torch.mean((y_pred - self.target_y)**2))
            except (ValueError, RuntimeError):
                return -PENALTY
        
        initial_mse = error_function(initial_constants)
        try:
            result = optimize.minimize(
                error_function, 
                initial_constants, 
                method='Nelder-Mead',
                options={'maxiter': MAXITER_OPT, 'adaptive': True}
            )
            
            final_mse = result.fun

            if result.success and final_mse < initial_mse:
                self.constants = [round(c, CONST_DECIMALS) for c in result.x]
                return True
        except Exception:
            self.constants = initial_constants
            return False

        self.constants = initial_constants
        return False

# -------------------------------------------------
# POLICY NETWORK (unverändert)
# -------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size - 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, seq):
        t = seq if isinstance(seq, torch.Tensor) else torch.LongTensor(seq)
        if t.dim() == 1: t = t.unsqueeze(0)
        embedded = self.embed(t)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))
    
    def select_action(self, seq, valid_actions=None):
        with torch.no_grad():
            logits = self.forward(seq).squeeze(0)
            if valid_actions is not None and valid_actions:
                mask = torch.full_like(logits, float('-inf'))
                mask[valid_actions] = 0
                dist = torch.distributions.Categorical(logits=logits + mask)
            else:
                dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item() + 1, dist.log_prob(action)

# -------------------------------------------------
# REPLAY BUFFER (unverändert)
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    def add(self, *args): self.buffer.append(args)
    def sample(self, batch_size): return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self): return len(self.buffer)

# -------------------------------------------------
# TRAINER (MODIFIZIERT für Curriculum Learning)
# -------------------------------------------------
class DSPTrainer:
    def __init__(self, target_fn, allowed_operators=None, episodes=EPISODES):
        # NEU: Initialisierung der Curriculum-Parameter
        self.current_max_length = CURRICULUM_START_LENGTH
        
        # MODIFIZIERT: Environment wird mit der Start-Länge initialisiert
        self.env = SymbolicRegressionEnv(target_fn, allowed_operators, max_length=self.current_max_length)
        
        self.model = PolicyNetwork(self.env.vocab_size)
        self.target_model = PolicyNetwork(self.env.vocab_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer()
        self.episodes = episodes
        self.steps_done = 0
        self.best_expr, self.best_reward = None, float('-inf')

        # NEU: Historie für die Überprüfung der Stagnation
        self.curriculum_reward_history = []
    
    def get_valid_actions(self):
        # Unverändert
        if self.env.required_operands <= 0: return []
        actions = []
        if self.env.required_operands > 1 or len(self.env.tokens) == 0:
            actions.extend(self.env.binary_ops)
            actions.extend(self.env.unary_ops)
        actions.extend(self.env.terminals)
        return [a - 1 for a in set(actions)]

    def select_action(self, state):
        # Unverändert
        self.steps_done += 1
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        valid_actions = self.get_valid_actions()
        if not valid_actions: return None, None
        if random.random() < epsilon:
            return random.choice(valid_actions) + 1, None
        return self.model.select_action(state, valid_actions)
    
    def optimize_model(self):
        # Unverändert
        if len(self.memory) < BATCH_SIZE: return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # MODIFIZIERT: Padding erfolgt jetzt dynamisch auf die aktuelle maximale Länge im Batch
        state_tensor = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(s) for s in states], batch_first=True, padding_value=0)
        next_state_tensor = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(s) for s in next_states], batch_first=True, padding_value=0)
        action_tensor = torch.LongTensor([a - 1 for a in actions])
        reward_tensor = torch.FloatTensor(rewards)
        done_tensor = torch.FloatTensor(dones)
        
        current_q = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_state_tensor).max(1)[0]
            target_q = reward_tensor + GAMMA * next_q * (1 - done_tensor)
        
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def train(self):
        pbar = enhanced_progress_bar(range(1, self.episodes + 1), f"Training (max_len={self.current_max_length})")
        for episode in pbar:
            state = self.env.reset()
            done = False
            while not done:
                action, log_prob = self.select_action(state)
                if action is None: break
                
                next_state, reward, done, info = self.env.step(action)
                
                # MODIFIZIERT: Umgang mit optimierten Konstanten
                final_reward = reward
                if done and info.get('expr'):
                    improved = self.env.optimize_constants()
                    if improved:
                        # Neuberechnung des Rewards nach der Optimierung
                        final_reward = self.env._calculate_reward() 
                        info['expr'] = self.env.get_expression_str()
                    
                    self.curriculum_reward_history.append(final_reward)

                    if final_reward > self.best_reward:
                        self.best_reward = final_reward
                        self.best_expr = self.simplify_expression(info['expr'])
                        mse_val = calculate_mse(self.env.target_fn, self.best_expr)
                        pbar.set_description(f"Best MSE: {mse_val:.4f} (max_len={self.current_max_length})")
                        if mse_val < 1e-6:
                           print_highlighted(f"\nPerfekte Approximation erreicht: {self.best_expr} (MSE: {mse_val:.6f})")
                           return self.best_expr, self.best_reward
                
                self.memory.add(state, action, next_state, final_reward, done)
                state = next_state

            if episode % POLICY_UPDATE_FREQ == 0: self.optimize_model()
            if episode % TARGET_UPDATE_FREQ == 0: self.target_model.load_state_dict(self.model.state_dict())

            # NEU: Curriculum Learning Logik
            if episode % CURRICULUM_UPDATE_FREQ == 0 and self.current_max_length < MAX_LENGTH_LIMIT:
                self._update_curriculum()

        return self.best_expr, self.best_reward

    def _update_curriculum(self):
        # NEU: Hilfsfunktion zur Überprüfung und Durchführung des Curriculum-Updates
        if len(self.curriculum_reward_history) < CURRICULUM_UPDATE_FREQ:
            return # Nicht genügend Daten für eine Entscheidung

        # Nimm die Rewards der aktuellen Lernphase
        recent_rewards = self.curriculum_reward_history
        split_point = len(recent_rewards) // 2
        
        # Vergleiche die Performance der ersten und zweiten Hälfte der Phase
        mean_first_half = np.mean(recent_rewards[:split_point])
        mean_second_half = np.mean(recent_rewards[split_point:])
        
        improvement = (mean_second_half - mean_first_half) / (abs(mean_first_half) + 1e-8)

        # Wenn die Verbesserung unter dem Schwellenwert liegt -> Stagnation -> Komplexität erhöhen
        if improvement < CURRICULUM_STAGNATION_THRESHOLD:
            self.current_max_length = min(self.current_max_length + CORRICLUM_STEP, MAX_LENGTH_LIMIT)
            self.env.max_length = self.current_max_length
            
            # WICHTIG: Buffer leeren, da sich die Zustandsrepräsentation (Padding) ändert
            self.memory.buffer.clear() 
            
            print_highlighted(f"\n[Curriculum Update] Stagnation erkannt. Erhöhe MAX_LENGTH auf {self.current_max_length}.", color=Fore.YELLOW)
        
        # Historie für die nächste Lernphase zurücksetzen
        self.curriculum_reward_history = []
    
    def simplify_expression(self, expr_str):
        # Unverändert
        if not expr_str: return None
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str.replace('^', '**'))
            simplified_expr = expr.xreplace({n: round(n, CONST_DECIMALS) for n in expr.atoms(sp.Number)})
            return str(sp.expand(simplified_expr)).replace('**', '^')
        except Exception:
            return expr_str

# -------------------------------------------------
# MAIN & HELPER FUNCTIONS (unverändert)
# -------------------------------------------------
# In MAIN & HELPER FUNCTIONS
def evaluate_safely(expr_str, x_val):
    """
    MODIFIZIERT: Wertet einen Ausdruck sicher aus, indem die 'eval'-Umgebung
    korrekt eingeschränkt wird, um den AttributeError zu vermeiden.
    """
    try:
        safe_expr = expr_str.replace('^', '**')

        # Definiere die sichere Umgebung mit allen erlaubten Funktionen und Konstanten
        safe_dict = {
            "x": x_val,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "pi": np.pi,
            "e": np.e  # Kleines 'e' ist standard und vermeidet Konflikte mit 1e-5 etc.
        }
        
        # KORREKTE Methode zur Absicherung: Übergib ein leeres __builtins__-Wörterbuch
        safe_globals = {"__builtins__": {}}
        
        val = eval(safe_expr, safe_globals, safe_dict)
        
        # Verhindere komplexe Zahlen und gib immer einen float zurück
        return float(val) if not isinstance(val, complex) else np.nan
        
    except Exception:  # Fange alle möglichen Fehler von eval ab
        return np.nan


def calculate_mse(target_fn, expr_str, x_range=X_RANGE, n_points=200):
    if not expr_str: return float('inf')
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    y_true = np.array([target_fn(x) for x in x_vals])
    y_pred = np.array([evaluate_safely(expr_str, x) for x in x_vals])
    
    valid_mask = ~np.isnan(y_pred)
    if not np.any(valid_mask): return float('inf')
    
    return np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)

def plot_result(target_fn, expr_str, title="Function Approximation"):
    x_vals = np.linspace(X_RANGE[0], X_RANGE[1], 200)
    y_true = np.array([target_fn(x) for x in x_vals])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_true, '-', color='black', label="Target Function", linewidth=2.5)
    
    if expr_str:
        y_pred = np.array([evaluate_safely(expr_str, x) for x in x_vals])
        valid_mask = ~np.isnan(y_pred)
        mse = calculate_mse(target_fn, expr_str)
        label_text = f"Approximation: {expr_str}\n(MSE = {mse:.5f})"
        plt.plot(x_vals[valid_mask], y_pred[valid_mask], '--', color='red', label=label_text, linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def run_experiment(target_function_tuple):
    name, fn, ops = target_function_tuple
    print_highlighted(f"\n{'='*20} Optimizing target: {name} {'='*20}", color=Fore.CYAN)
    
    # Der Trainer wird wie gewohnt initialisiert
    trainer = DSPTrainer(fn, allowed_operators=ops)
    
    # trainer.train() gibt IMMER nur 2 Werte zurück
    best_expr, best_reward = trainer.train()
    
    # Wir berechnen den MSE hier in der Wrapper-Funktion, um den fehlenden Wert zu erhalten
    mse = calculate_mse(fn, best_expr)
    
    # Die Ausgabe der einzelnen Ergebnisse bleibt gleich
    print_highlighted(f"\n--- ERGEBNIS FÜR: {name} ---", color=Fore.GREEN)
    print(f"→ Bester gefundener Ausdruck: {best_expr}")
    print(f"→ Finaler Reward: {best_reward:.4f}, Finaler MSE: {mse:.6f}")
    plot_result(fn, best_expr, title=f"Approximation für '{name}'")
    
    # WICHTIG: Gib hier explizit und immer die vier erwarteten Werte zurück
    return name, best_expr, best_reward, mse

# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
if __name__ == "__main__":
    # Definiere die Ziel-Funktionen, die getestet werden sollen
    TARGET_FUNCTIONS = [
        ("sin(x)", lambda x: math.sin(x), ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("sin(x) + 0.5*x", lambda x: math.sin(x) + 0.5*x, ['+', '-', '*', 'sin', 'cos']),
        ("exp(-x) * cos(x)", lambda x: math.exp(-x) * math.cos(x), ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
        ("x^2 + 2*x + 1", lambda x: x**2 + 2*x + 1, ['+', '-', '*', '^']),
        ("x^3", lambda x: x**3, ['+', '-', '*', '^']),
        ("x^3 - 3*x^2 + 2*x - 1", lambda x: x**3 - 3*x**2 + 2*x - 1, ['+', '-', '*', '^']),
        ("2*x^3 + x^2 - x + 5", lambda x: 2*x**3 + x**2 - x + 5, ['+', '-', '*', '^']),
        ("x^4", lambda x: x**4, ['+', '-', '*', '^']),
        ("3*x^4 + 2*x^3 - x^2 + x - 2", lambda x: 3*x**4 + 2*x**3 - x**2 + x - 2, ['+', '-', '*', '^']),
        ("x^5", lambda x: x**5, ['+', '-', '*', '^']),
        ("x^5 - 2*x^4 + x^3 - x + 2", lambda x: x**5 - 2*x**4 + x**3 - x + 2, ['+', '-', '*', '^']),
    ]

    # Speichere die Ergebnisse in einer Liste für eine geordnete Ausgabe
    results_list = []
    for func_tuple in TARGET_FUNCTIONS:
        # Die run_experiment Funktion gibt bereits ein 4-Tupel zurück
        result_tuple = run_experiment(func_tuple)
        results_list.append(result_tuple)

    # Zusammenfassung aller Läufe
    print("\n" + "="*30 + " ZUSAMMENFASSUNG ALLER LÄUFE " + "="*30)
    
    # Bereite die Daten für die Tabelle vor
    summary_data_for_table = []
    # Iteriere durch die Ergebnisliste und entpacke das 4-Tupel
    for name, expr, reward, mse in results_list:
        # Stelle sicher, dass der Ausdruck nicht None ist, bevor er zur Tabelle hinzugefügt wird
        expr_to_display = expr if expr is not None else "Keine Lösung gefunden"
        summary_data_for_table.append([name, expr_to_display, f"{reward:.4f}", f"{mse:.6f}"])
        
    # Gib die Tabelle mit den bekannten, alten Spaltenüberschriften aus
    print_table(summary_data_for_table, headers=["Zielfunktion", "Gefundener Ausdruck", "Reward", "MSE"])