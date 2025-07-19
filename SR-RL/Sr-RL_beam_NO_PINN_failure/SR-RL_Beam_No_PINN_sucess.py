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
import seaborn as sns

sys.setrecursionlimit(10000)

# -------------------------------------------------
# Problem­parameter DGL (Balken)
# -------------------------------------------------
E = 150e9   # Elastizitätsmodul in Pa
I = 4e-6    # Flächenträgheitsmoment in m^4
q = 60e3    # Gleichlast in N/m
L = 2.5     # Balkenlänge in m
bc_weight = 1 # Gewichtung der Randbedingungen im Verlust

# -------------------------------------------------
# Wichtige Hyperparameter RL & Symbolic Regression
# -------------------------------------------------
MAX_LENGTH    = 17    # Etwas mehr Länge für das Polynom 4. Grades
X_RANGE       = (0, L)
N_POINTS      = 50
EPISODES      = 20000
BATCH_SIZE    = 64
LR            = 0.0001
EMBED_SIZE    = 64
HIDDEN_SIZE   = 128
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.1
EPSILON_DECAY = 20000
MEMORY_SIZE   = 5000
POLICY_UPDATE_FREQ = 5
TARGET_UPDATE_FREQ = 50
PROB_LOG_FREQ = 1000

# Sicherheits-/Stabilitätsparameter
GRAD_CLIP     = 1.0
BONUS_SCALE   = 0.1

# Konstantenoptimierung
CONST_DECIMALS = 5 # Höhere Genauigkeit für die Koeffizienten
CONST_RANGE = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, L]
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
# ENVIRONMENT mit Konstantenoptimierung
# -------------------------------------------------
class SymbolicRegressionEnv:
    def __init__(self, allowed_operators=None, x_range=X_RANGE, n_points=N_POINTS, max_length=MAX_LENGTH):
        self.x_range, self.n_points, self.max_length = x_range, n_points, max_length
        self.TOK_PAD, self.TOK_X, self.TOK_CONST = 0, 1, 2
        self.TOK_PLUS, self.TOK_MINUS, self.TOK_MUL, self.TOK_DIV, self.TOK_POW = 3, 4, 5, 6, 7
        self.TOK_SIN, self.TOK_COS, self.TOK_EXP = 8, 9, 10
        self.op_mapping = {'+': 3, '-': 4, '*': 5, '/': 6, '^': 7, 'sin': 8, 'cos': 9, 'exp': 10}
        self.token_str = {v: k for k, v in self.op_mapping.items()}
        self.token_str.update({0: "PAD", 1: "x", 2: "C"})
        if allowed_operators:
            self.binary_ops = [self.op_mapping[op] for op in allowed_operators if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in allowed_operators if op in ['sin', 'cos', 'exp']]
        else:
            self.binary_ops = [self.op_mapping[op] for op in self.op_mapping if op in ['+', '-', '*', '/', '^']]
            self.unary_ops = [self.op_mapping[op] for op in self.op_mapping if op in ['sin', 'cos', 'exp']]
        self.terminals = [self.TOK_X, self.TOK_CONST]
        self.vocab_size = len(self.token_str)
        self.reset()

    def reset(self):
        self.tokens, self.constants = [], []
        self.required_operands, self.steps, self.done = 1, 0, False
        self.x_values_np = np.linspace(self.x_range[0], self.x_range[1], self.n_points)
        self.prev_normalized_loss = None
        return [self.TOK_PAD] * self.max_length

    def step(self, action_token):
        if self.done: raise RuntimeError("step() called after episode is done")
        self.steps += 1
        if self.tokens and self.tokens[-1] == self.TOK_POW and action_token == self.TOK_X:
            self.done = True; return self._pad_obs(), -10.0, True, {'expr': None}
        if action_token == self.TOK_CONST: self.constants.append(random.choice(CONST_RANGE))
        self.tokens.append(action_token)
        if action_token in self.binary_ops: self.required_operands += 1
        elif action_token not in self.unary_ops: self.required_operands -= 1
        reward, info = 0.0, {}
        if self.required_operands == 0 or self.steps >= self.max_length:
            self.done = True
            expr_str = self.get_expression_str()
            if self.required_operands == 0 and expr_str and 'x' in expr_str:
                try:
                    if sp.simplify(sp.sympify(expr_str.replace('^','**'))).is_Number:
                        reward, info['expr'] = -10.0, None
                    else:
                        reward, info['expr'] = self._calculate_reward(expr_str), expr_str
                except Exception: reward, info['expr'] = -10.0, None
            else: reward, info['expr'] = -10.0, None
        return self._pad_obs(), reward, self.done, info

    def _pad_obs(self):
        return self.tokens + [self.TOK_PAD] * (self.max_length - len(self.tokens))

    def _get_full_expr_str(self, const_override=None):
        try:
            temp_constants = const_override if const_override is not None else self.constants
            const_idx = 0
            def build(i):
                nonlocal const_idx
                if i >= len(self.tokens):
                    raise IndexError("Incomplete expression tree.")
                tok = self.tokens[i]
                if tok in self.binary_ops:
                    l, next_i = build(i+1); r, next_i = build(next_i)
                    return f"({l} {self.token_str[tok]} {r})", next_i
                elif tok in self.unary_ops:
                    arg, next_i = build(i+1); return f"{self.token_str[tok]}({arg})", next_i
                elif tok == self.TOK_X: return "x", i+1
                elif tok == self.TOK_CONST:
                    val = temp_constants[const_idx]; const_idx += 1
                    return f"({val})", i+1
            expr, _ = build(0)
            return expr
        except (IndexError, ValueError, TypeError):
            return None

# Ersetze die Funktion in SymbolicRegressionEnv

    def _calculate_physics_loss(self, const_override=None):
        try:
            expr_str = self._get_full_expr_str(const_override)
            if expr_str is None: return 1e10

            x_sym = sp.Symbol('x')
            u_sym = sp.sympify(expr_str.replace('^', '**'))

            # Benötigte Ableitungen für die dimensionslose DGL
            u_fourth_sym = sp.diff(u_sym, x_sym, 4) # Für die DGL u'''' = 1
            u_double_sym = sp.diff(u_sym, x_sym, 2) # Für die Randbedingungen u''(0), u''(1)

            # Numerische Funktionen erstellen
            u_func = sp.lambdify(x_sym, u_sym, 'numpy')
            u_dd_func = sp.lambdify(x_sym, u_double_sym, 'numpy')
            u_xxxx_func = sp.lambdify(x_sym, u_fourth_sym, 'numpy')

            # WICHTIG: Wir rechnen jetzt auf dem dimensionslosen Bereich [0, 1]
            x_values_norm = np.linspace(0.0, 1.0, self.n_points)

            with np.errstate(all='ignore'):
                # 1. DGL-Residual berechnen: u'''' - 1 = 0
                u_xxxx_vals = u_xxxx_func(x_values_norm)
                # Das Residual ist jetzt viel einfacher und stabiler
                residual = u_xxxx_vals - 1.0

                # 2. Randbedingungen auswerten (am Rand des [0, 1] Intervalls)
                u0 = u_func(0.0)
                u1 = u_func(1.0) # bei x_norm = 1
                u0_dd = u_dd_func(0.0)
                u1_dd = u_dd_func(1.0) # bei x_norm = 1

            # Auf ungültige Werte prüfen (bleibt gleich)
            if u_xxxx_vals is None or np.any(np.isnan(residual)) or np.any(np.isinf(residual)) or \
               np.iscomplexobj(residual) or np.iscomplex(u0) or np.iscomplex(u1) or \
               np.iscomplex(u0_dd) or np.iscomplex(u1_dd) or math.isnan(u0) or math.isnan(u1) or \
               math.isnan(u0_dd) or math.isnan(u1_dd):
                return 1e10

            # 3. Verlust berechnen
            mse_res = float(np.mean(np.real(residual)**2))
            # Vier Randbedingungen auf dem [0, 1] Intervall
            bc_loss = u0**2 + u1**2 + u0_dd**2 + u1_dd**2

            # Der bc_weight kann jetzt viel effektiver sein. Ggf. sogar erhöhen, z.B. auf 10
            return mse_res + bc_weight * bc_loss
        except Exception:
            return 1e10

# Ersetze die Funktion in SymbolicRegressionEnv

    def _calculate_reward(self, expr_str): # expr_str ist hier nur ein Trigger
        total_loss = self._calculate_physics_loss()
        if total_loss >= 1e9: return -10.0
        
        # NEUE SKALIERUNG: Da der Loss jetzt viel kleiner ist, brauchen wir keine
        # aggressive Normalisierung mehr. 'total_loss' wirkt hier direkt als 'normalized'.
        # Die exp-Funktion ist eine gute Wahl für Losses um 0-10.
        base_reward = math.exp(-total_loss) 
        
        bonus = 0.0
        if self.prev_normalized_loss is not None:
            # Wir verwenden 'total_loss' hier, da es die neue normalisierte Metrik ist
            improvement = (self.prev_normalized_loss - total_loss) / self.prev_normalized_loss if self.prev_normalized_loss > 0 else 0.0
            bonus = BONUS_SCALE * max(0, min(improvement, 1))
        self.prev_normalized_loss = total_loss
        return base_reward + bonus

    def get_expression_str(self):
        return self._get_full_expr_str()

    def optimize_constants(self):
        const_count = self.tokens.count(self.TOK_CONST)
        if const_count == 0: return False
        initial_loss = self._calculate_physics_loss()
        if initial_loss >= 1e9: return False

        def error_func(const_values):
            return self._calculate_physics_loss(const_override=const_values)

        res = optimize.minimize(error_func, self.constants, method='L-BFGS-B', options={'maxiter': MAXITER_OPT})

        if res.success and res.fun < initial_loss:
            self.constants = [round(c, CONST_DECIMALS) for c in res.x]
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
        self.fc = nn.Linear(hidden_size, vocab_size - 1)
        nn.init.zeros_(self.embed.weight[0])
        for param in self.lstm.parameters():
            if param.dim() > 1: nn.init.xavier_uniform_(param)
            else: nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, seq):
        t = torch.LongTensor(seq) if not isinstance(seq, torch.Tensor) else seq
        if t.dim() == 1: t = t.unsqueeze(0)
        
        lengths = (t != self.embed.padding_idx).sum(dim=1)
        if (lengths == 0).any():
            out = torch.zeros(t.size(0), self.fc.out_features, device=t.device)
            if (lengths > 0).any():
                non_empty_mask = lengths > 0
                non_empty_t = t[non_empty_mask]
                non_empty_lengths = lengths[non_empty_mask]
                embedded = self.embed(non_empty_t)
                packed = nn.utils.rnn.pack_padded_sequence(embedded, non_empty_lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, (hidden, _) = self.lstm(packed)
                out[non_empty_mask] = self.fc(hidden.squeeze(0))
            return out.squeeze(0)

        embedded = self.embed(t)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden.squeeze(0)).squeeze(0)

    def select_action(self, seq, valid_actions=None):
        with torch.no_grad():
            logits = self.forward(seq)
            if valid_actions:
                mask = torch.full_like(logits, -float('inf'))
                if valid_actions:
                   mask[valid_actions] = 0.0
                logits += mask
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item() + 1, dist.log_prob(action), dist.entropy()

# -------------------------------------------------
# REPLAY BUFFER
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    def add(self, *args): self.buffer.append(args)
    def sample(self, n): return random.sample(self.buffer, min(n, len(self.buffer)))
    def __len__(self): return len(self.buffer)

# -------------------------------------------------
# TRAINER mit Integration der Optimierung und Analyse
# -------------------------------------------------
class DSPTrainer:
    def __init__(self, allowed_operators=None, episodes=EPISODES, lr=LR):
        self.env = SymbolicRegressionEnv(allowed_operators=allowed_operators)
        self.model = PolicyNetwork(self.env.vocab_size)
        self.target_model = PolicyNetwork(self.env.vocab_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.episodes = episodes
        self.batch_size = BATCH_SIZE
        self.steps_done = 0

        self.history, self.entropy_history = [], []
        self.action_probs_history, self.prob_log_epochs = [], []
        self.top_solutions = []
        self.seen_expressions = set()

    def get_valid_actions(self):
        required_operands = self.env.required_operands
        valid = []
        if required_operands >= 1 and len(self.env.tokens) < self.env.max_length - 2:
            valid.extend(self.env.binary_ops)
            valid.extend(self.env.unary_ops)
        if required_operands > 0:
            if self.env.tokens and self.env.tokens[-1] == self.env.TOK_POW:
                valid.append(self.env.TOK_CONST)
            else:
                valid.extend(self.env.terminals)
        return [a - 1 for a in list(set(valid)) if a > 0]

    def select_action(self, state):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        valid_actions = self.get_valid_actions()
        if not valid_actions: return None, None, None
        if random.random() < epsilon:
            return random.choice(valid_actions) + 1, None, None
        return self.model.select_action(state, valid_actions)

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        state_t = torch.LongTensor(states)
        action_t = torch.LongTensor([a - 1 for a in actions]).unsqueeze(1)
        reward_t = torch.FloatTensor(rewards)
        next_state_t = torch.LongTensor(next_states)
        done_t = torch.FloatTensor(dones)

        current_q = self.model(state_t).gather(1, action_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_state_t).max(1)[0].detach()
            target_q = reward_t + GAMMA * next_q * (1 - done_t)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def _update_top_solutions(self, reward, expr):
        if expr is None or expr in self.seen_expressions: return
        self.seen_expressions.add(expr)
        if len(self.top_solutions) < 5: self.top_solutions.append((reward, expr))
        elif reward > self.top_solutions[-1][0]:
            self.top_solutions.pop(); self.top_solutions.append((reward, expr))
        self.top_solutions.sort(key=lambda x: x[0], reverse=True)

    def train(self):
        for episode in enhanced_progress_bar(range(1, self.episodes + 1), desc="Training läuft", unit=" Ep."):
            state = self.env.reset()
            episode_entropies = []
            final_reward = 0
            info = {}

            while not self.env.done:
                action, _, entropy = self.select_action(state)
                if action is None: break
                if entropy is not None: episode_entropies.append(entropy.item())
                next_state, reward, done, info = self.env.step(action)
                final_reward = reward
                self.memory.add(state, action, next_state, reward, done)
                state = next_state

            expr = info.get('expr')
            if expr:
                if self.env.optimize_constants():
                    final_reward = self._calculate_reward_from_expr(expr)
                    expr = self.env.get_expression_str()

                simplified = self.simplify_expression(expr)
                self._update_top_solutions(final_reward, simplified)

            self.history.append(final_reward)
            self.entropy_history.append(np.mean(episode_entropies) if episode_entropies else 0)

            if episode % POLICY_UPDATE_FREQ == 0: self.optimize_model()
            if episode % TARGET_UPDATE_FREQ == 0: self.target_model.load_state_dict(self.model.state_dict())
            if episode % PROB_LOG_FREQ == 0: self._log_action_probs(episode)
            if episode % 1000 == 0:
                best_expr_so_far = self.top_solutions[0][1] if self.top_solutions else "None"
                print(f"\nEpisode {episode}/{self.episodes}, Avg reward: {np.mean(self.history[-1000:]):.3f}, Best: {best_expr_so_far}")

        return self.top_solutions
    
    def _calculate_reward_from_expr(self, expr):
        """ Recalculates reward based on the current (optimized) constants. """
        # This function directly calls the environment's reward calculation
        # It's a bit redundant but makes the training loop clearer
        return self.env._calculate_reward(expr)


    def _log_action_probs(self, episode):
        self.prob_log_epochs.append(episode)
        prob_samples = []
        with torch.no_grad():
            # Leerer Zustand
            empty_state = [self.env.TOK_PAD] * self.env.max_length
            logits = self.model.forward(empty_state)
            probs = F.softmax(logits, dim=0).cpu().numpy()
            prob_samples.append(probs)

            # Zustand mit einem Token (falls binäre Operatoren vorhanden)
            if self.env.binary_ops:
                single_state = [self.env.binary_ops[0]] + [self.env.TOK_PAD] * (self.env.max_length - 1)
                logits = self.model.forward(single_state)
                probs = F.softmax(logits, dim=0).cpu().numpy()
                prob_samples.append(probs)

        # Durchschnitt der Wahrscheinlichkeiten über verschiedene Zustände
        avg_probs = np.mean(prob_samples, axis=0) if prob_samples else np.zeros(self.env.vocab_size - 1)
        self.action_probs_history.append(avg_probs)

    def simplify_expression(self, expr_str):
        if not expr_str: return None
        try:
            # expand() führt die Multiplikationen aus und vereinfacht den Term zu einer reinen Summe
            simplified_expr = sp.expand(sp.sympify(expr_str.replace('^', '**')))
            return str(simplified_expr).replace('**', '^')
        except Exception:
            return expr_str

# -------------------------------------------------
# MAIN-FUNKTION UND ANALYSE
# -------------------------------------------------
def analytic_solution(x_vals):
    # Deflection is positive/downwards
    return (q / (24 * E * I)) * (x_vals**4 - 2 * L * x_vals**3 + L**3 * x_vals)

def moving_average(data, window_size):
    if len(data) < window_size: return []
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_reward_history(history, run_name):
    plt.figure(figsize=(12, 6))
    plt.plot(history, label='Reward pro Episode', alpha=0.5)
    mv_avg = moving_average(history, 100)
    if len(mv_avg) > 0:
        plt.plot(np.arange(99, len(history)), mv_avg, 'r', label='Gleitender Durchschnitt (100 Ep.)')
    plt.title(f'Reward-Verlauf - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Finaler Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_entropy_history(entropy_history, run_name):
    plt.figure(figsize=(12, 6))
    plt.plot(entropy_history, label='Durchschnittliche Entropie', alpha=0.6)
    mv_avg = moving_average(entropy_history, 100)
    if len(mv_avg) > 0:
        plt.plot(np.arange(99, len(entropy_history)), mv_avg, 'g', label='Gleitender Durchschnitt (100 Ep.)')
    plt.title(f'Policy-Entropie-Verlauf - {run_name}')
    plt.xlabel('Episode')
    plt.ylabel('Entropie')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_action_prob_heatmap(action_probs, epochs, env, run_name):
    if not action_probs: return
    action_labels = [env.token_str.get(i + 1, f"UNK_{i+1}") for i in range(env.vocab_size - 1)]
    prob_array = np.array(action_probs)

    if prob_array.shape[0] > 1:
        plt.figure(figsize=(14, 8))
        sns.heatmap(prob_array, cmap='viridis', xticklabels=action_labels,
                   yticklabels=[f"Ep {ep}" for ep in epochs], cbar_kws={'label': 'Durchschnittliche Wahrscheinlichkeit'})
        plt.title(f'Heatmap der Aktionswahrscheinlichkeiten - {run_name}')
        plt.xlabel('Aktion (Token)')
        plt.ylabel('Episode')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Nicht genügend Daten für Heatmap in {run_name}")

def plot_result(expr_str, run_name):
    # Die analytische Lösung bleibt im Original-Koordinatensystem
    x_np = np.linspace(X_RANGE[0], X_RANGE[1], 200)
    y_analytic = analytic_solution(x_np)
    
    y_pred = np.full_like(x_np, np.nan)
    if expr_str:
        try:
            # u_func ist die gefundene dimensionslose Lösung u_bar(x_bar)
            u_bar_func = sp.lambdify(sp.Symbol('x'), sp.sympify(expr_str.replace('^', '**')), 'numpy')
            
            # Um zu plotten, skalieren wir das Ergebnis zurück
            x_bar_np = x_np / L                  # x -> x_bar
            u_bar_pred = u_bar_func(x_bar_np)    # u_bar = f(x_bar)
            
            # Charakteristische Durchbiegung zum Zurückskalieren
            u_c = (q * L**4) / (E * I)
            y_pred = u_bar_pred * u_c           # u = u_bar * u_c

        except Exception as e:
            print(f"Fehler bei der Auswertung des Ausdrucks für den Plot: {e}")

    plt.figure(figsize=(12, 7))
    plt.plot(x_np, y_analytic, 'k-', label='Analytische Lösung', linewidth=3)
    plt.plot(x_np, y_pred, 'r--', label=f"Gefundene Lösung: {expr_str}", linewidth=2)
    plt.title(f'Vergleich der Lösungen - {run_name}')
    plt.xlabel('$x$ [m]')
    plt.ylabel('Durchbiegung $u(x)$ [m]')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    error = np.abs(y_pred - y_analytic)
    plt.figure(figsize=(12, 5))
    plt.plot(x_np, error, 'b-')
    plt.title(f'Absoluter Fehler - {run_name}')
    plt.xlabel('$x$ [m]')
    plt.ylabel('Absoluter Fehler [m]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def calculate_mse(expr_str):
    if not expr_str: return float('inf')
    try:
        # 1. Erstelle die x-Werte und die exakte, dimensionale Lösung
        x_np = np.linspace(X_RANGE[0], X_RANGE[1], 200)
        y_analytic = analytic_solution(x_np)

        # 2. Erstelle die Funktion für die gefundene, dimensionslose Lösung u_bar(x_bar)
        u_bar_func = sp.lambdify(sp.Symbol('x'), sp.sympify(expr_str.replace('^', '**')), 'numpy')

        # 3. Skaliere das Ergebnis zurück, um die dimensionale Vorhersage y_pred zu erhalten
        x_bar_np = x_np / L                  # Konvertiere x in x_bar
        u_bar_pred = u_bar_func(x_bar_np)    # Berechne u_bar
        
        u_c = (q * L**4) / (E * I)           # Berechne den Skalierungsfaktor
        y_pred = u_bar_pred * u_c            # Skaliere u_bar zurück zu u (y_pred)

        # 4. Prüfe auf ungültige Werte und berechne den MSE
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
             return float('inf')
        return np.mean((y_analytic - y_pred)**2)

    except Exception:
        return float('inf')

def run_experiment(allowed_operators=None, episodes=EPISODES, run_name=""):
    print_highlighted(f"\n=== Starte Lauf: {run_name} ===", Fore.CYAN)
    trainer = DSPTrainer(allowed_operators=allowed_operators, episodes=episodes)
    top_solutions = trainer.train()

    print_highlighted(f"\n--- Analysediagramme für Lauf: {run_name} ---", Fore.YELLOW)
    plot_reward_history(trainer.history, run_name)
    plot_entropy_history(trainer.entropy_history, run_name)
    if trainer.action_probs_history:
        plot_action_prob_heatmap(trainer.action_probs_history, trainer.prob_log_epochs, trainer.env, run_name)
    
    return top_solutions

if __name__ == "__main__":
    EXPERIMENT_CONFIGS = [
        ("Lauf 1 (Balken): Lineare Operatoren", ['+', '-', '*']),
        ("Lauf 2 (Balken): Polynom-Operatoren", ['+', '-', '*', '^']),
    ]
    results = {}

    print_highlighted("Parameter des physikalischen Problems (Balken):", Fore.YELLOW)
    param_data = [
        ["Elastizitätsmodul (E)", f"{E:.2e} Pa"],
        ["Flächenträgheitsmoment (I)", f"{I:.2e} m^4"],
        ["Gleichlast (q)", f"{q:.2e} N/m"],
        ["Balkenlänge (L)", f"{L} m"]
    ]
    print_table(param_data)

    for name, operators in EXPERIMENT_CONFIGS:
        top_solutions = run_experiment(allowed_operators=operators, episodes=EPISODES, run_name=name)
        results[name] = []
        if top_solutions:
            for reward, expr in top_solutions:
                mse = calculate_mse(expr)
                results[name].append({'expr': expr, 'reward': reward, 'mse': mse})
            plot_result(results[name][0]['expr'], name)

    print("\n\n" + "="*80)
    print_highlighted("=== ZUSAMMENFASSUNG ALLER LÄUFE ===", Fore.MAGENTA)
    print("="*80)
    for name, solutions in results.items():
        print(f"\n--- {name} ---")
        if solutions:
            # Sortiere nach MSE für eine bessere Vergleichbarkeit der Lösungsqualität
            solutions.sort(key=lambda s: s['mse'])
            table_data = [[f"#{i+1}", s['expr'], f"{s['reward']:.4f}", f"{s['mse']:.2e}"] for i, s in enumerate(solutions)]
            print(tabulate(table_data, headers=["Rang (nach MSE)", "Gefundener Ausdruck", "Reward", "MSE vs. Analytisch"], tablefmt="grid"))
        else:
            print("Keine gültige Lösung in diesem Lauf gefunden.")
    print("\n" + "="*80)