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
    """Definiert das vollständige dimensionale Balkenproblem."""
    def __init__(self, E: float, I: float, L: float, q_func: Callable[[float], float], bcs: List[BoundaryCondition]):
        self.E = E
        self.I = I
        self.L = L
        self.q_func = q_func
        self.bcs = bcs
        self.EI = E * I

class NondimensionalProblem:
    """Leitet das dimensionslose Problem aus dem dimensionalen ab."""
    def __init__(self, problem: ProblemDefinition, Q_scale: float):
        self.problem = problem
        self.L = problem.L
        self.Q_scale = Q_scale
        self.U = (Q_scale * self.L**4) / self.problem.EI

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
def perform_integration(x_array: np.ndarray, y_vals: np.ndarray) -> List[np.ndarray]:
    """Integriert eine Funktion viermal numerisch."""
    integrals = []
    current_integral = y_vals
    
    for _ in range(4):
        next_integral = np.zeros_like(x_array)
        for i in range(1, len(x_array)):
            dx = x_array[i] - x_array[i-1]
            next_integral[i] = next_integral[i-1] + 0.5 * (current_integral[i] + current_integral[i-1]) * dx
        integrals.append(next_integral)
        current_integral = next_integral
        
    return integrals

# -----------------------------------------------------------------------------
# Optimierer-Klassen
# -----------------------------------------------------------------------------
class NondimensionalOptimizer:
    """Findet C1-C4 für das dimensionslose Balkenproblem."""
    def __init__(self, nondim_problem: NondimensionalProblem):
        self.nondim_problem = nondim_problem
        self.xi_eval = np.linspace(0, 1, GLOBAL_PARAMS['xi_eval_points'])
        f_vals = nondim_problem.nondim_load_shape(self.xi_eval)
        self.integrals = perform_integration(self.xi_eval, f_vals) # [G1, G2, G3, G4]

    def optimize_constants(self) -> np.ndarray:
        def objective(params):
            C1, C2, C3, C4 = params
            bc_error = 0.0
            G2_interp = lambda xi: np.interp(xi, self.xi_eval, self.integrals[1])
            G4_interp = lambda xi: np.interp(xi, self.xi_eval, self.integrals[3])

            for order, xi0, value in self.nondim_problem.bc_nondim:
                if order == 0:
                    u_hat_bc = G4_interp(xi0) + C1*xi0**3/6 + C2*xi0**2/2 + C3*xi0 + C4
                    bc_error += (u_hat_bc - value)**2
                elif order == 2:
                    u_hat_prime2_bc = G2_interp(xi0) + C1*xi0 + C2
                    bc_error += (u_hat_prime2_bc - value)**2
            return bc_error

        result = minimize(objective, np.zeros(4), method='L-BFGS-B')
        return result.x

class DimensionalOptimizer:
    """Findet D1-D4 für das dimensionale Balkenproblem."""
    def __init__(self, problem: ProblemDefinition):
        self.problem = problem
        self.x_eval = np.linspace(0, problem.L, GLOBAL_PARAMS['xi_eval_points'])
        q_vals = np.vectorize(problem.q_func)(self.x_eval)
        self.integrals = perform_integration(self.x_eval, q_vals) # [Int_q, Int2_q, Int3_q, Int4_q]

    def optimize_constants(self) -> np.ndarray:
        def objective(params):
            D1, D2, D3, D4 = params
            bc_error = 0.0
            EI = self.problem.EI
            Int2_q_interp = lambda x: np.interp(x, self.x_eval, self.integrals[1])
            Int4_q_interp = lambda x: np.interp(x, self.x_eval, self.integrals[3])
            
            for bc in self.problem.bcs:
                if bc.order == 0:
                    u_bc = (1/EI) * (Int4_q_interp(bc.x0) + D1*bc.x0**3/6 + D2*bc.x0**2/2 + D3*bc.x0 + D4)
                    bc_error += (u_bc - bc.value)**2
                elif bc.order == 2:
                    u_prime2_bc = (1/EI) * (Int2_q_interp(bc.x0) + D1*bc.x0 + D2)
                    bc_error += (u_prime2_bc - bc.value)**2
            return bc_error

        result = minimize(objective, np.zeros(4), method='L-BFGS-B')
        return result.x

# -----------------------------------------------------------------------------
# Hauptfunktion zur Ausführung eines Szenarios
# -----------------------------------------------------------------------------
def run_scenario(scenario_name: str, params: dict):
    """Führt ein komplettes Szenario aus und gibt die Ergebnisse aus."""
    E, I, L, q0 = params['E'], params['I'], params['L'], params['q0']

    # Problem definieren
    q_func = lambda x: -q0
    bcs = [
        BoundaryCondition(order=0, x0=0.0, value=0.0), BoundaryCondition(order=2, x0=0.0, value=0.0),
        BoundaryCondition(order=0, x0=L,   value=0.0), BoundaryCondition(order=2, x0=L,   value=0.0)
    ]
    problem = ProblemDefinition(E, I, L, q_func, bcs)
    nondim_problem = NondimensionalProblem(problem, Q_scale=q0)

    # Analytische Lösungen
    U = nondim_problem.U
    analytical_func_dim = lambda x: -(q0 / (24 * problem.EI)) * (x**4 - 2*L*x**3 + L**3*x)
    C_an = np.array([0.5, 0.0, -1.0/24.0, 0.0])
    D_an = np.array([q0*L/2, 0, -q0*L**3/24, 0])

    # Ausgabe der Formeln und Werte
    print("\n" + "="*80)
    print(f" Szenario: {scenario_name.upper()} ".center(80, "="))
    print("="*80)
    print("\n📝 Formeln und Definitionen (Balken 4. Ordnung):")
    print(f"  - Dimensionale DGL:       EIu''''(x) = -q₀")
    print(f"  - Dimensionslose DGL:     d⁴û/dξ⁴ = -1")
    print(f"  - Skalierungsfaktor U:      U = (q₀ * L⁴) / (EI)")
    
    # Lösen
    print("\n🚀 Lösungsversuche:")
    nondim_optimizer = NondimensionalOptimizer(nondim_problem)
    C_opt = nondim_optimizer.optimize_constants()
    print(f"  - Nicht-dimensional gefunden: C₁={C_opt[0]:.4f}, C₂={C_opt[1]:.4f}, C₃={C_opt[2]:.4f}, C₄={C_opt[3]:.4f}")
    print(f"  - Analytische C-Werte:     C₁={C_an[0]:.4f}, C₂={C_an[1]:.4f}, C₃={C_an[2]:.4f}, C₄={C_an[3]:.4f}")

    dim_optimizer = DimensionalOptimizer(problem)
    D_opt = dim_optimizer.optimize_constants()
    print(f"  - Dimensional gefunden:       D₁={D_opt[0]:.3e}, D₂={D_opt[1]:.3e}, D₃={D_opt[2]:.3e}, D₄={D_opt[3]:.3e}")
    print(f"  - Analytische D-Werte:     D₁={D_an[0]:.3e}, D₂={D_an[1]:.3e}, D₃={D_an[2]:.3e}, D₄={D_an[3]:.3e}")
    
    # Auswerten und Plotten
    x_plot = np.linspace(0, L, GLOBAL_PARAMS['plot_points'])
    xi_plot = x_plot / L
    
    u_analytical = analytical_func_dim(x_plot)
    
    G4_vals_nondim = nondim_optimizer.integrals[3]
    u_nondim_scaled = U * (G4_vals_nondim + C_opt[0]*xi_plot**3/6 + C_opt[1]*xi_plot**2/2 + C_opt[2]*xi_plot + C_opt[3])
    
    Int4_q_vals_dim = dim_optimizer.integrals[3]
    u_dim_direct = (1/problem.EI) * (Int4_q_vals_dim + D_opt[0]*x_plot**3/6 + D_opt[1]*x_plot**2/2 + D_opt[2]*x_plot + D_opt[3])

    error_nondim = np.max(np.abs(u_nondim_scaled - u_analytical))
    error_dim = np.max(np.abs(u_dim_direct - u_analytical))
    
    print("\n📊 Ergebnisse und Fehler:")
    print(f"  - Max. Fehler (Nicht-dimensional): {error_nondim:.4e} m")
    print(f"  - Max. Fehler (Dimensional):       {error_dim:.4e} m")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=GLOBAL_PARAMS['figure_size'])
    plt.plot(x_plot, -u_analytical, 'b-', label='Analytische Lösung', linewidth=GLOBAL_PARAMS['line_width_analytical'], zorder=5)
    plt.plot(x_plot, -u_nondim_scaled, 'r--', label=f'Nicht-dimensional (Fehler: {error_nondim:.2e})', linewidth=GLOBAL_PARAMS['line_width_methods'])
    plt.plot(x_plot, -u_dim_direct, 'g:', label=f'Dimensional (Fehler: {error_dim:.2e})', linewidth=GLOBAL_PARAMS['line_width_methods'])

    plt.title(f'Balkendurchbiegung - {scenario_name}')
    plt.xlabel('Position x [m]')
    plt.ylabel('Durchbiegung u(x) [m] (positiv nach unten)')
    plt.legend()
    plt.gca().invert_yaxis()
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
    favorable_params = {'E': 1000.0, 'I': 0.1, 'L': 10.0, 'q0': 5.0}
    challenging_params = {'E': 2.1e11, 'I': 1.943e-5, 'L': 8.0, 'q0': 10000.0}

    run_scenario("Günstige Parameter", favorable_params)
    run_scenario("Anspruchsvolle Parameter", challenging_params)
    