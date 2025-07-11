

from pysr import install as pysr_install, PySRRegressor
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp

# einmalig, wenn noch nicht geschehen
pysr_install()

# =============================================================================
# Configuration: domain, number of points, and target functions
# =============================================================================
X_RANGE = (0.0, 5.0)
N_POINTS = 50

# List of tuples: (Name, String expression, allowed operators)
TARGET_FUNCTIONS = [
    ("sin(x)", "sin(x)", ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
    ("sin(x) + 0.5*x", "sin(x) + 0.5*x", ['+', '-', '*', '^', 'sin', 'cos']),
    ("exp(-x) * cos(x)", "exp(-x) * cos(x)", ['+', '-', '*', '^', 'sin', 'cos', 'exp']),
    ("x^2 + 2*x + 1", "x^2 + 2*x + 1", ['+', '-', '*', '^']),
    ("x^3", "x^3", ['+', '-', '*', '^']),
    ("x^3 - 3*x^2 + 2*x - 1", "x^3 - 3*x^2 + 2*x - 1", ['+', '-', '*', '^']),
    ("2*x^3 + x^2 - x + 5", "2*x^3 + x^2 - x + 5", ['+', '-', '*', '^']),
    ("x^4", "x^4", ['+', '-', '*', '^']),
    ("3*x^4 + 2*x^3 - x^2 + x - 2", "3*x^4 + 2*x^3 - x^2 + x - 2", ['+', '-', '*', '^']),
    ("x^5", "x^5", ['+', '-', '*', '^']),
    ("x^5 - 2*x^4 + x^3 - x + 2", "x^5 - 2*x^4 + x^3 - x + 2", ['+', '-', '*', '^']),
]

def make_lambda_from_str(expr_str: str):
    """
    Convert an expression string like "sin(x) + 0.5*x" into a Python function f(x).
    We replace '^' by '**' and eval in a restricted namespace.
    """
    expr_py = expr_str.replace('^', '**')
    def f(x):
        return eval(expr_py,
                    {"__builtins__": {}},
                    {"sin": math.sin, "cos": math.cos, "exp": math.exp, "x": x})
    return f

def make_dataset(target_lambda, x_range=X_RANGE, n_points=N_POINTS):
    """
    Generate X, y arrays for PySR. 
    X: shape (n_points, 1), y: shape (n_points,)
    """
    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.array([target_lambda(x) for x in xs], dtype=float)
    return xs.reshape(-1,1), ys

def map_ops_to_pysr(ops_list):
    """
    Map a list like ['+', '-', '*', '^', 'sin'] to PySR binary_operators/unary_operators.
    PySR expects 'pow' for exponentiation and names 'sin','cos','exp' for unary.
    """
    binary_map = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "pow"}
    unary_map  = {"sin": "sin", "cos": "cos", "exp": "exp"}
    bin_ops = []
    un_ops = []
    for op in ops_list:
        if op in binary_map:
            bin_ops.append(binary_map[op])
        if op in unary_map:
            un_ops.append(unary_map[op])
    # If empty, pass None so PySR uses defaults
    if not bin_ops:
        bin_ops = None
    if not un_ops:
        un_ops = None
    return bin_ops, un_ops

# Disable interactive showing of figures until the end
plt.ioff()

# =============================================================================
# Main loop: for each target function, run PySR, collect expression & MSE, prepare plots
# =============================================================================
def run_pysr_for_targets():
    results = {}
    for name, expr_str, ops in TARGET_FUNCTIONS:
        print(f"\n=== PySR für Ziel: {name} ===")
        # Create Python lambda for numeric target
        target_lambda = make_lambda_from_str(expr_str)
        # Make dataset
        X, y = make_dataset(target_lambda)
        # Map operators
        bin_ops, un_ops = map_ops_to_pysr(ops)
        # Instantiate PySRRegressor
        try:
            model = PySRRegressor(
                niterations=40,
                population_size=50,       # ensure tournament_selection_n < population_size
                binary_operators=bin_ops,
                unary_operators=un_ops,
                model_selection="best",
                loss="L2DistLoss()",      # L2 loss
                verbosity=1,
                # variable_names default is ["x0"] for single feature
                # random_seed=0,          # optionally fix seed for reproducibility
            )
            model.fit(X, y)
        except Exception as e:
            print(f"Fehler beim Fit für {name}: {e}")
            continue

        # Retrieve best expression
        best = model.get_best()
        expr_pysr = best.get("sympy_format", None)
        # Simplify expression via Sympy
        simplified_expr = None
        if expr_pysr is not None:
            try:
                sym_expr = sp.sympify(expr_pysr)
                simp = sp.simplify(sym_expr)
                simplified_expr = str(simp)
            except Exception as e:
                print(f"Sympy-Vereinfachung fehlgeschlagen für {name}: {e}")
                simplified_expr = expr_pysr
        # Compute MSE on training set
        try:
            y_pred_train = model.predict(X)
            mse = float(np.mean((y - y_pred_train)**2))
        except Exception:
            mse = float('nan')
        print(f"→ Gefundener Ausdruck: {expr_pysr}")
        if simplified_expr is not None:
            print(f"→ Vereinfachter Ausdruck: {simplified_expr}")
        print(f"→ MSE auf Trainingsdaten: {mse:.6g}")
        results[name] = dict(expr=expr_pysr, simplified=simplified_expr, mse=mse, model=model, target_lambda=target_lambda)

        # Prepare Plot: target vs. PySR approximation
        xs_plot = np.linspace(X_RANGE[0], X_RANGE[1], 200)
        # a) target curve
        y_true_plot = np.array([target_lambda(xv) for xv in xs_plot], dtype=float)

        # b) PySR prediction via model.predict if possible
        try:
            X_plot = xs_plot.reshape(-1,1)
            y_pred_plot = model.predict(X_plot)
            y_pred_plot = np.array(y_pred_plot, dtype=float)
        except Exception:
            # Fallback: eval the expr_pysr string, defining x0 in namespace
            y_pred_list = []
            if expr_pysr is not None:
                for xv in xs_plot:
                    try:
                        val = eval(expr_pysr,
                                   {"__builtins__": {}},
                                   {"sin": math.sin, "cos": math.cos, "exp": math.exp, "x0": xv})
                    except Exception:
                        val = float('nan')
                    y_pred_list.append(val)
                y_pred_plot = np.array(y_pred_list, dtype=float)
            else:
                y_pred_plot = np.full_like(xs_plot, np.nan, dtype=float)

        # c) Create figure and plot, but do not show yet
        fig = plt.figure(figsize=(6,4))
        plt.plot(xs_plot, y_true_plot, '-', label="Ziel", lw=2)
        mask = np.isfinite(y_pred_plot)
        label_expr = simplified_expr or expr_pysr or ""
        if mask.any() and label_expr:
            plt.plot(xs_plot[mask], y_pred_plot[mask], '--', label=f"PySR ≈ {label_expr}", lw=2)
        elif mask.any():
            plt.plot(xs_plot[mask], y_pred_plot[mask], '--', label="PySR Approximation", lw=2)
        else:
            print("Hinweis: Alle Vorhersage-Werte ungültig (NaN/Inf). Kein Plot für Approximation.")
        plt.title(f"{name} | MSE={mse:.4g}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        # Figure bleibt offen; plt.show() wird später aufgerufen

    # Summary
    print("\n=== Zusammenfassung PySR-Ergebnisse ===")
    for name, info in results.items():
        expr = info.get('expr')
        simplified = info.get('simplified')
        expr_str_for_print = str(simplified) if simplified is not None else str(expr)
        mse = info.get('mse')
        print(f"{name:<30} | Expr: {expr_str_for_print:<30} | MSE: {mse:.6g}")
    return results

if __name__ == "__main__":
    results = run_pysr_for_targets()
    # Nach allen Fits: alle vorbereiteten Figuren auf einmal anzeigen
    plt.show()