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
    'plot_points': 200
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

    # 1. Problem definieren
    q_func = lambda x: q0 * x
    bc1 = BoundaryCondition(order=0, x0=0.0, value=0.0)
    bc2 = BoundaryCondition(order=1, x0=L, value=0.0)
    problem = ProblemDefinition(E, A, L, q_func, [bc1, bc2])
    
    Q_scale = q0 * L
    nondim_problem = NondimensionalProblem(problem, Q_scale)

    # Analytische Lösung
    analytical_func_nondim = lambda xi: -xi**3/6 + 0.5*xi
    U = nondim_problem.U
    analytical_func_dim = lambda x: U * analytical_func_nondim(x / L)

    # --- Ausgabe der Formeln und Werte ---
    print("\n" + "="*80)
    print(f" Szenario: {scenario_name.upper()} ".center(80, "="))
    print("="*80)
    print("\n📝 Formeln und Definitionen:")
    print(f"  - Dimensionale Last:      q(x) = q₀ * x   (wobei q₀ die Steigung in N/m² ist)")
    print(f"  - Dimensionslose Last:    f(ξ) = q(Lξ)/(q₀*L) = ξ")
    # ... (weitere Ausgaben bleiben gleich)

    print("\n🔢 Berechnete Parameter für dieses Szenario:")
    print(f"  - E = {E:.2e} Pa, A = {A:.2e} m², L = {L:.2e} m")
    print(f"  - Last-Steigung q₀ = {q0:.2e} N/m²")
    print(f"  - Maximale Last q(L) = {(q0*L):.2e} N/m")
    print(f"  - Q_scale = {Q_scale:.2e} N/m")
    print(f"  - U = {U:.6f} m")
    
    # 2. Lösen mit beiden Methoden
    print("\n🚀 Lösungsversuche:")
    nondim_optimizer = NondimensionalOptimizer(nondim_problem)
    C1_opt, C2_opt = nondim_optimizer.optimize_constants()
    print(f"  - Nicht-dimensional gefunden: C₁={C1_opt:.6f}, C₂={C2_opt:.6f} (Analytisch: C₁=0.5, C₂=0.0)")

    D1_rescaled = Q_scale * L * C1_opt
    D2_rescaled = Q_scale * L**2 * C2_opt
    D1_analytical = Q_scale * L * 0.5
    D2_analytical = Q_scale * L**2 * 0.0
    print(f"  - N-D zurückskaliert:         D₁={D1_rescaled:.4e}, D₂={D2_rescaled:.4e}")

    dim_optimizer = DimensionalOptimizer(problem)
    D1_opt, D2_opt = dim_optimizer.optimize_constants()
    print(f"  - Dimensional gefunden:       D₁={D1_opt:.4e}, D₂={D2_opt:.4e} (Analytisch: D₁={D1_analytical:.4e}, D₂={D2_analytical:.4e})")

    # 3. Auswerten und Plotten
    x_plot = np.linspace(0, L, GLOBAL_PARAMS['plot_points'])
    xi_plot = x_plot / L
    
    u_analytical = analytical_func_dim(x_plot)
    u_nondim_scaled = U * (-nondim_optimizer.G_vals + C1_opt * xi_plot + C2_opt)
    u_dim_direct = (1/problem.EA) * (-dim_optimizer.Int_q_vals + D1_opt * x_plot + D2_opt)
    
    error_nondim = np.max(np.abs(u_nondim_scaled - u_analytical))
    error_dim = np.max(np.abs(u_dim_direct - u_analytical))
    
    print("\n📊 Ergebnisse und Fehler:")
    print(f"  - Max. Fehler (Nicht-dimensional): {error_nondim:.4e} m")
    print(f"  - Max. Fehler (Dimensional):       {error_dim:.4e} m")
    print(f"  - Max. analytische Verschiebung:   {np.max(u_analytical):.4e} m")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=GLOBAL_PARAMS['figure_size'])
    plt.plot(x_plot, u_analytical, 'b-', label='Analytische Lösung', linewidth=GLOBAL_PARAMS['line_width_analytical'], zorder=5)
    plt.plot(x_plot, u_nondim_scaled, 'r--', label=f'Nicht-dimensional (Fehler: {error_nondim:.2e})', linewidth=GLOBAL_PARAMS['line_width_methods'])
    plt.plot(x_plot, u_dim_direct, 'g:', label=f'Dimensional (Fehler: {error_dim:.2e})', linewidth=GLOBAL_PARAMS['line_width_methods'])
    
    plt.title(f'Vergleich der Lösungen - {scenario_name}')
    plt.xlabel('Position x [m]')
    plt.ylabel('Verschiebung u(x) [m]')
    plt.legend()
    plt.grid(alpha=GLOBAL_PARAMS['plot_alpha'])
    plt.show()

    # --- NEU: ZUSÄTZLICHER PLOT FÜR DEN FEHLERVERLAUF ---
    error_nondim_array = u_nondim_scaled - u_analytical
    error_dim_array = u_dim_direct - u_analytical
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=GLOBAL_PARAMS['figure_size'])
    plt.plot(x_plot, error_nondim_array, 'r-', label=f'Fehler (Nicht-dimensional)')
    plt.plot(x_plot, error_dim_array, 'g-', label=f'Fehler (Dimensional)')
    
    plt.title(f'Fehlerverlauf der Methoden - {scenario_name}')
    plt.xlabel('Position x [m]')
    plt.ylabel('Absoluter Fehler [m]')
    plt.legend()
    plt.grid(alpha=GLOBAL_PARAMS['plot_alpha'])
    plt.show()


# -----------------------------------------------------------------------------
# Szenarien definieren und ausführen
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    favorable_params = {
        'E': 100.0,
        'A': 1.0,
        'L': 5.0,
        'q0': 2.0
    }
    
    challenging_params = {
        'E': 2.1e11,
        'A': 0.001,
        'L': 20.0,
        'q0': 2500.0
    }

    run_scenario("Günstige Parameter", favorable_params)
    run_scenario("Anspruchsvolle Parameter", challenging_params)
    
    print("\n" + "="*80)
    print(" Analyse der Ergebnisse ".center(80, "="))
    print("="*80)