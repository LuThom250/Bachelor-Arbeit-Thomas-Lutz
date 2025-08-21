import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable, List, Tuple

# -----------------------------------------------------------------------------
# Globale Einstellungen
# -----------------------------------------------------------------------------
GLOBAL_PARAMS = {
    'figure_size': (10, 7),
    'line_width_analytical': 4,
    'line_width_methods': 2,
    'plot_alpha': 0.3,
    'xi_eval_points': 200,
    'plot_points': 200,
    # NEU: Schriftgrößen zentral steuern
    'fontsize_title': 24,
    'fontsize_labels': 22,
    'fontsize_legend': 20
}

# -----------------------------------------------------------------------------
# Klassen-Definitionen
# -----------------------------------------------------------------------------
class BoundaryCondition:
    """Definiert eine einzelne Randbedingung."""
    def __init__(self, order: int, x0: float, value: float):
        self.order = order
        self.x0 = x0
        self.value = value

class ProblemDefinition:
    """Definiert das vollständige dimensionale Problem."""
    def __init__(self, E: float, A: float, L: float, q_func: Callable[[float], float], bcs: List[BoundaryCondition]):
        self.E = E
        self.A = A
        self.L = L
        self.q_func = q_func
        self.bcs = bcs
        self.EA = E * A

class NondimensionalProblem:
    """Leitet das dimensionslose Problem aus dem dimensionalen ab."""
    def __init__(self, problem: ProblemDefinition, Q_scale: float):
        self.problem = problem
        self.L = problem.L
        self.Q_scale = Q_scale
        self.U = (Q_scale * self.L**2) / self.problem.EA

        self.bc_nondim: List[Tuple[int, float, float]] = []
        for bc in problem.bcs:
            xi0 = bc.x0 / self.L
            value_nondim = bc.value * (self.L ** bc.order) / self.U
            self.bc_nondim.append((bc.order, xi0, value_nondim))

    def nondim_load_shape(self, xi: np.ndarray) -> np.ndarray:
        """Berechnet die dimensionslose Last f(xi)."""
        x = xi * self.L
        q_vals = np.vectorize(self.problem.q_func)(x)
        return q_vals / self.Q_scale

# -----------------------------------------------------------------------------
# Hilfsfunktionen für die Integration
# -----------------------------------------------------------------------------
def compute_double_integral(xi_array: np.ndarray, f_vals: np.ndarray) -> np.ndarray:
    """Integriert die dimensionslose Lastfunktion f(xi) zweimal."""
    Q = np.zeros_like(xi_array)
    for i in range(1, len(xi_array)):
        Q[i] = Q[i-1] + 0.5 * (f_vals[i] + f_vals[i-1]) * (xi_array[i] - xi_array[i-1])
    G = np.zeros_like(xi_array)
    for i in range(1, len(xi_array)):
        G[i] = G[i-1] + 0.5 * (Q[i] + Q[i-1]) * (xi_array[i] - xi_array[i-1])
    return G

def compute_double_integral_dim(x_array: np.ndarray, q_func: Callable) -> np.ndarray:
    """Integriert die dimensionale Lastfunktion q(x) zweimal."""
    q_vals = np.vectorize(q_func)(x_array)
    Q = np.zeros_like(x_array)
    for i in range(1, len(x_array)):
        Q[i] = Q[i-1] + 0.5 * (q_vals[i] + q_vals[i-1]) * (x_array[i] - x_array[i-1])
    G = np.zeros_like(x_array)
    for i in range(1, len(x_array)):
        G[i] = G[i-1] + 0.5 * (Q[i] + Q[i-1]) * (x_array[i] - x_array[i-1])
    return G

# -----------------------------------------------------------------------------
# Optimierer-Klassen
# -----------------------------------------------------------------------------
class NondimensionalOptimizer:
    """Findet C1 und C2 für das dimensionslose Problem."""
    def __init__(self, nondim_problem: NondimensionalProblem):
        self.nondim_problem = nondim_problem
        self.xi_eval = np.linspace(0, 1, GLOBAL_PARAMS['xi_eval_points'])
        self.f_vals = nondim_problem.nondim_load_shape(self.xi_eval)
        self.G_vals = compute_double_integral(self.xi_eval, self.f_vals)

    def optimize_constants(self) -> Tuple[float, float]:
        def objective(params):
            C1, C2 = params
            bc_error = 0.0
            for order, xi0, value in self.nondim_problem.bc_nondim:
                if order == 0:
                    G_at_xi0 = np.interp(xi0, self.xi_eval, self.G_vals)
                    u_hat_bc = -G_at_xi0 + C1 * xi0 + C2
                    bc_error += (u_hat_bc - value)**2
                elif order == 1:
                    Q_vals = np.gradient(self.G_vals, self.xi_eval)
                    Q_at_xi0 = np.interp(xi0, self.xi_eval, Q_vals)
                    u_hat_prime_bc = -Q_at_xi0 + C1
                    bc_error += (u_hat_prime_bc - value)**2
            return bc_error

        result = minimize(objective, [0.0, 0.0], method='L-BFGS-B')
        return result.x if result.success else (0.0, 0.0)

class DimensionalOptimizer:
    """Findet D1 und D2 für das dimensionale Problem."""
    def __init__(self, problem: ProblemDefinition):
        self.problem = problem
        self.x_eval = np.linspace(0, self.problem.L, GLOBAL_PARAMS['xi_eval_points'])
        self.Int_q_vals = compute_double_integral_dim(self.x_eval, self.problem.q_func)

    def optimize_constants(self) -> Tuple[float, float]:
        def objective(params):
            D1, D2 = params
            bc_error = 0.0
            EA = self.problem.EA
            for bc in self.problem.bcs:
                if bc.order == 0:
                    Int_q_at_x0 = np.interp(bc.x0, self.x_eval, self.Int_q_vals)
                    u_bc = (1/EA) * (-Int_q_at_x0 + D1 * bc.x0 + D2)
                    bc_error += (u_bc - bc.value)**2
                elif bc.order == 1:
                    Int_q_prime_vals = np.gradient(self.Int_q_vals, self.x_eval)
                    Int_q_prime_at_x0 = np.interp(bc.x0, self.x_eval, Int_q_prime_vals)
                    u_prime_bc = (1/EA) * (-Int_q_prime_at_x0 + D1)
                    bc_error += (u_prime_bc - bc.value)**2
            return bc_error

        result = minimize(objective, [0.0, 0.0], method='L-BFGS-B')
        return result.x if result.success else (0.0, 0.0)

# -----------------------------------------------------------------------------
# Hauptfunktion zur Ausführung eines Szenarios
# -----------------------------------------------------------------------------
def run_scenario(scenario_name: str, params: dict):
    """Führt ein komplettes Szenario aus und gibt die Ergebnisse aus."""
    E, A, L, q0 = params['E'], params['A'], params['L'], params['q0']

    q_func = lambda x: q0 * x
    bc1 = BoundaryCondition(order=0, x0=0.0, value=0.0)
    bc2 = BoundaryCondition(order=1, x0=L, value=0.0)
    problem = ProblemDefinition(E, A, L, q_func, [bc1, bc2])
    
    Q_scale = q0 * L
    nondim_problem = NondimensionalProblem(problem, Q_scale)

    analytical_func_nondim = lambda xi: -xi**3/6 + 0.5*xi
    U = nondim_problem.U
    analytical_func_dim = lambda x: U * analytical_func_nondim(x / L)

    print(f"\n--- Scenario: {scenario_name} ---")
    nondim_optimizer = NondimensionalOptimizer(nondim_problem)
    C1_opt, C2_opt = nondim_optimizer.optimize_constants()

    dim_optimizer = DimensionalOptimizer(problem)
    D1_opt, D2_opt = dim_optimizer.optimize_constants()

    x_plot = np.linspace(0, L, GLOBAL_PARAMS['plot_points'])
    xi_plot = x_plot / L
    
    u_analytical = analytical_func_dim(x_plot)
    u_nondim_scaled = U * (-nondim_optimizer.G_vals + C1_opt * xi_plot + C2_opt)
    u_dim_direct = (1/problem.EA) * (-dim_optimizer.Int_q_vals + D1_opt * x_plot + D2_opt)
    
    error_nondim = np.max(np.abs(u_nondim_scaled - u_analytical))
    error_dim = np.max(np.abs(u_dim_direct - u_analytical))

    # ========= Plot 1 =========
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=GLOBAL_PARAMS['figure_size'])
    plt.plot(x_plot, u_analytical, 'b-', label='Analytical', 
             linewidth=GLOBAL_PARAMS['line_width_analytical'], alpha=0.6, zorder=1)
    plt.plot(x_plot, u_nondim_scaled, 'r--', label=f'Non-dimensional (error: {error_nondim:.2e})', 
             linewidth=GLOBAL_PARAMS['line_width_methods'], zorder=3)
    plt.plot(x_plot, u_dim_direct, 'g:', label=f'Dimensional (error: {error_dim:.2e})', 
             linewidth=GLOBAL_PARAMS['line_width_methods'], zorder=3)
    plt.title(f'Comparison of solutions - {scenario_name}', fontsize=GLOBAL_PARAMS['fontsize_title'])
    plt.xlabel('Position x [m]', fontsize=GLOBAL_PARAMS['fontsize_labels'])
    plt.ylabel('Displacement u(x) [m]', fontsize=GLOBAL_PARAMS['fontsize_labels'])
    plt.legend(fontsize=GLOBAL_PARAMS['fontsize_legend'])
    plt.grid(alpha=GLOBAL_PARAMS['plot_alpha'])
    plt.show()

    # ========= Plot 2 =========
    error_nondim_array = u_nondim_scaled - u_analytical
    error_dim_array = u_dim_direct - u_analytical
    plt.figure(figsize=GLOBAL_PARAMS['figure_size'])
    plt.plot(x_plot, error_nondim_array, 'r-', label='Error (non-dimensional)')
    plt.plot(x_plot, error_dim_array, 'g-', label='Error (dimensional)')
    plt.title(f'Error over x - {scenario_name}', fontsize=GLOBAL_PARAMS['fontsize_title'])
    plt.xlabel('Position x [m]', fontsize=GLOBAL_PARAMS['fontsize_labels'])
    plt.ylabel('Absolute error [m]', fontsize=GLOBAL_PARAMS['fontsize_labels'])
    plt.legend(fontsize=GLOBAL_PARAMS['fontsize_legend'])
    plt.grid(alpha=GLOBAL_PARAMS['plot_alpha'])
    plt.show()

# -----------------------------------------------------------------------------
# Szenarien ausführen
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    favorable_params = {'E': 100.0, 'A': 1.0, 'L': 5.0, 'q0': 2.0}
    challenging_params = {'E': 2.1e11, 'A': 0.001, 'L': 20.0, 'q0': 2500.0}
    run_scenario("Favorable parameters", favorable_params)
    run_scenario("Challenging parameters", challenging_params)