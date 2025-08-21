import random
import operator
import warnings

import numpy as np
import sympy as sp
from deap import base, creator, tools, gp, algorithms
import matplotlib.pyplot as plt

# ----------------------------
# Hyperparameter-Sektion
# Passe hier alle wichtigen Einstellungen zentral an
# ----------------------------
# Physik / ODE-Parameter
E = 100.0             # Elastizitätsmodul
A = 1.0               # Querschnittsfläche
q0 = 10.0             # Lastparameter
L = 5.0               # Länge des Stabes

# Fitness-Evaluation / Grid
GRID_POINTS = 100     # Anzahl der Auswertungspunkte in (0, L)
BC_THRESHOLD = 1e3    # Grenze für Boundary-Condition-Werte, oberhalb verwerfen
LOSS_HIGH = 1e6       # Sehr großer Verlust, wenn Ausdruck unbrauchbar

# Boundary-Condition-Penalty
BC_PENALTY_WEIGHT = 20.0  # Gewicht der BC-Penalty relativ zur Residual-MSE

# GP-Einstellungen
POP_SIZE = 100        # Populationsgröße
N_GEN = 30            # Anzahl Generationen (bei __main__ verwendet)
CX_PROB = 0.5         # Crossover-Wahrscheinlichkeit
MUT_PROB = 0.2       # Mutations-Wahrscheinlichkeit
TOURNEY_SIZE = 3      # Tournament-Größe für Selektion

# Initialbaum-Generierung
INIT_MIN_DEPTH = 1    # Minimale Tiefe für Ramp-Half-and-Half
INIT_MAX_DEPTH = 3    # Maximale Tiefe für Ramp-Half-and-Half

# Baumgrößen-Limits (um Bloat einzudämmen)
MAX_TREE_HEIGHT = 7   # Maximale Baumhöhe
MAX_TREE_NODES = 12   # Maximale Anzahl Knoten im Baum

# Bereich für Zufallskonstanten
RAND_CONST_MIN = -5
RAND_CONST_MAX = 5

# ----------------------------
# Suppress noisy runtime warnings global (wir fangen später gezielt ab)
# ----------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------
# Sympy Symbol
# ----------------------------
x = sp.Symbol('x')

# ----------------------------
# DEAP GP setup: primitives operating on Sympy expressions
# ----------------------------
def add_sym(a, b):
    return a + b

def sub_sym(a, b):
    return a - b

def mul_sym(a, b):
    return a * b

def div_sym(a, b):
    # symbolische Division, numerisch können inf/NaN entstehen, später penalisiert
    return a / b

def pow_sym(a, b):
    # symbolische Potenz; riskant, wenn b nicht-integer oder groß, aber wir fangen später ab
    return a ** b

# Optional trig/exponential:
def sin_sym(a):
    return sp.sin(a)

def cos_sym(a):
    return sp.cos(a)

def exp_sym(a):
    return sp.exp(a)

# ----------------------------
# PrimitiveSet: 1 Argument (x)
# ----------------------------
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')

# Binäre Primitives
pset.addPrimitive(add_sym, 2, name="add")
pset.addPrimitive(sub_sym, 2, name="sub")
pset.addPrimitive(mul_sym, 2, name="mul")
pset.addPrimitive(div_sym, 2, name="div")
pset.addPrimitive(pow_sym, 2, name="pow")

# Wenn trig/expo erwünscht, auskommentieren:
# pset.addPrimitive(sin_sym, 1, name="sin")
# pset.addPrimitive(cos_sym, 1, name="cos")
# pset.addPrimitive(exp_sym, 1, name="exp")

# Ephemeral constants: benannte Funktion für Picklability
def gen_rand_const():
    return sp.Float(random.uniform(RAND_CONST_MIN, RAND_CONST_MAX))
pset.addEphemeralConstant("rand_const", gen_rand_const)

# Terminal 'x' ist per Default enthalten

# ----------------------------
# DEAP Fitness und Individual
# ----------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimierung
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Initial-Bäume: ramped half-and-half, Tiefe INIT_MIN_DEPTH bis INIT_MAX_DEPTH
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ----------------------------
# Caching: Sympy-Derivate & Lambdify
# Key: str(sym_expr), Value: (f_u, f_u1, f_u2, u_expr, u_prime) oder None bei ungültig
# ----------------------------
_eval_cache = {}

# ----------------------------
# Fitness-Evaluation: PDE-Residual + BC-Penalty
# ----------------------------
def evalPDE(individual):
    """
    Evaluiere Fitness eines Individuums:
    - Baue sympy-Expression u(x) aus GP-Baum
    - Berechne u' und u'' symbolisch
    - Lambdify zu numpy-Funktionen
    - Residual A*E * u''(x) + q0*x auf Grid in (0, L) mitteln (MSE)
    - BC: u(0)=0, u'(L)=0, mit Penalty
    Rückgabe: (loss,) für DEAP
    """
    # 1) Kompilieren auf Sympy-Expression
    try:
        func = gp.compile(individual, pset)
        sym_expr = func(x)
        if not isinstance(sym_expr, sp.Expr):
            sym_expr = sp.sympify(sym_expr)
    except Exception:
        return (LOSS_HIGH,)

    expr_key = str(sym_expr)
    if expr_key not in _eval_cache:
        try:
            u_expr = sym_expr
            u_prime = sp.diff(u_expr, x)
            u_double = sp.diff(u_expr, x, 2)
            f_u = sp.lambdify(x, u_expr, "numpy")
            f_u1 = sp.lambdify(x, u_prime, "numpy")
            f_u2 = sp.lambdify(x, u_double, "numpy")
            _eval_cache[expr_key] = (f_u, f_u1, f_u2, u_expr, u_prime)
        except Exception:
            _eval_cache[expr_key] = None

    funcs = _eval_cache[expr_key]
    if funcs is None:
        return (LOSS_HIGH,)
    f_u, f_u1, f_u2, u_expr, u_prime = funcs

    # 2) Numerische Evaluation auf Grid in (0, L), vermeide exakt 0/L
    xs = np.linspace(1e-6, L - 1e-6, GRID_POINTS)
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            u_vals = f_u(xs)
            u1_vals = f_u1(xs)
            u2_vals = f_u2(xs)
    except Exception:
        return (LOSS_HIGH,)

    # Prüfe gültige Arrays
    if not (isinstance(u_vals, np.ndarray) and isinstance(u2_vals, np.ndarray)):
        return (LOSS_HIGH,)
    # Prüfe NaN/Inf
    if np.any(~np.isfinite(u_vals)) or np.any(~np.isfinite(u2_vals)):
        return (LOSS_HIGH,)

    # PDE residual: A*E * u''(x) + q0 * x
    residual = A * E * u2_vals + q0 * xs
    mse_res = float(np.mean(residual ** 2))

    # 3) Boundary Conditions
    # BC1: u(0)=0
    try:
        # Versuche symbolisch
        u0_sym = u_expr.subs(x, 0)
        u0 = float(u0_sym)
    except Exception:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                u0 = float(f_u(0.0))
        except Exception:
            return (LOSS_HIGH,)

    # BC2: u'(L)=0
    try:
        uL1_sym = u_prime.subs(x, L)
        uL1 = float(uL1_sym)
    except Exception:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                uL1 = float(f_u1(L))
        except Exception:
            return (LOSS_HIGH,)

    # Falls BC-Werte zu groß, verwerfe früh
    if not np.isfinite(u0) or abs(u0) > BC_THRESHOLD or not np.isfinite(uL1) or abs(uL1) > BC_THRESHOLD:
        return (LOSS_HIGH,)

    # Nun safely quadrieren (Werte sind moderat)
    try:
        bc_pen = u0**2 + uL1**2
    except OverflowError:
        return (LOSS_HIGH,)

    # 4) Kombiniere Loss
    scale = np.mean((q0 * xs)**2) + 1e-8
    loss = (mse_res + BC_PENALTY_WEIGHT * bc_pen) / scale
    # Wenn loss NaN/Inf (ungewöhnlich), verwerfe
    if not np.isfinite(loss):
        return (LOSS_HIGH,)
    return (loss,)

toolbox.register("evaluate", evalPDE)

# ----------------------------
# Genetic operators
# ----------------------------
toolbox.register("select", tools.selTournament, tournsize=TOURNEY_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limit Baum-Höhe und Baum-Größe (Anzahl Knoten) um Bloat einzudämmen
# Beschränke Höhe
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
# Beschränke Anzahl Knoten (len(ind) gibt Knotenanzahl)
toolbox.decorate("mate", gp.staticLimit(key=lambda ind: len(ind), max_value=MAX_TREE_NODES))
toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: len(ind), max_value=MAX_TREE_NODES))

# ----------------------------
# Main GP run
# ----------------------------
def run_gp(pop_size=POP_SIZE, n_generations=N_GEN, verbose=False):
    """
    Führe GP aus. Gibt (Population, Logbook, HallOfFame) zurück.
    """
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("std", np.std)
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=CX_PROB, mutpb=MUT_PROB,
                                   ngen=n_generations,
                                   stats=stats, halloffame=hof,
                                   verbose=verbose)
    return pop, log, hof

# ----------------------------
# Utility: Sympy-Simplify-Ausgabe
# ----------------------------
def simplify_sympy(expr_str):
    try:
        expr = sp.sympify(expr_str)
        simp = sp.simplify(expr)
        return str(simp)
    except Exception:
        return expr_str

# ----------------------------
# Plot-Vergleich (ohne Glättung)
# ----------------------------
def plot_comparison(best_expr, L=L):
    """
    Plot: Analytische Lösung vs GP-Lösung (roh).
    """
    # Analytische Lösung
    u_analytic = -q0/(6*E*A) * x**3 + q0*L**2/(2*E*A) * x
    f_analytic = sp.lambdify(x, u_analytic, "numpy")

    xs_full = np.linspace(0.0, L, 200)
    try:
        f_best = sp.lambdify(x, best_expr, "numpy")
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            y_best = f_best(xs_full)
    except Exception:
        print("Konnte beste GP-Expression nicht numerisch auswerten.")
        return

    mask = np.isfinite(y_best)
    if not np.any(mask):
        print("GP-Ausdruck liefert keine finiten Werte; kein Plot.")
        return

    y_analytic = f_analytic(xs_full)

    plt.figure(figsize=(8, 5))
    plt.plot(xs_full, y_analytic, label="Analytische Lösung", linewidth=2)
    plt.plot(xs_full, y_best, label="GP-Lösung", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Vergleich: GP vs Analytische Lösung")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    # Reproduzierbarkeit
    random.seed(42)
    np.random.seed(42)

    print("Starting GP for rod ODE: A*E*u''(x) + q0*x = 0, BC: u(0)=0, u'(L)=0")
    pop, log, hof = run_gp(pop_size=POP_SIZE, n_generations=N_GEN, verbose=False)

    print("\n=== Top candidates ===")
    for ind in hof:
        try:
            expr = gp.compile(ind, pset)(x)
            expr_str = str(expr)
            simplified = simplify_sympy(expr_str)
            fitness = evalPDE(ind)[0]
            print(f"Raw expr:    {expr_str}")
            print(f"Simplified:  {simplified}")
            print(f"Loss:        {fitness:.6g}")
            print("-" * 40)
        except Exception:
            print("Fehler beim Anzeigen eines HallOfFame-Individuums.")

    # Vergleich mit analytischer Lösung und Ausgabe der besten GP-Lösung
    if len(hof) > 0:
        best_ind = hof[0]
        try:
            best_expr = gp.compile(best_ind, pset)(x)
            print("Analytische Lösung: u(x) = -q0/(6*E*A)*x^3 + q0*L^2/(2*E*A)*x")
            # Zeige die beste gefundene GP-Lösung symbolisch darunter:
            print("Beste GP-Lösung (symbolisch):")
            print(f"    u_GP(x) = {best_expr}")
            # Numerische MSE-Berechnung
            f_best = sp.lambdify(x, best_expr, "numpy")
            xs = np.linspace(0.0, L, 100)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                y_best = f_best(xs)
                y_analytic = sp.lambdify(x, -q0/(6*E*A) * x**3 + q0*L**2/(2*E*A) * x, "numpy")(xs)
            mask = np.isfinite(y_best)
            if np.any(mask):
                mse_vs_analytic = np.mean((y_best[mask] - y_analytic[mask])**2)
                print(f"MSE GP vs analytisch: {mse_vs_analytic:.6g}")
            else:
                print("GP-Lösung liefert keine finiten Werte; keine MSE.")
            plot_comparison(best_expr, L=L)
        except Exception as e:
            print("Konnte GP-Lösung nicht vergleichen/plotten:", e)
