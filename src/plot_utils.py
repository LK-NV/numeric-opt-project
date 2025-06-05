import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable, Optional, List, Tuple

FunctionEvaluationResult = Tuple[float, np.ndarray, Optional[np.ndarray]]
Constraint = Callable[[np.ndarray, bool], FunctionEvaluationResult]

# -------------------------------------------------------------
# QP  –  single 3-D scene: level-sets + simplex + central path
# -------------------------------------------------------------
def plot_central_path_qp(path: List[np.ndarray],
                         f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
                         save_as: str = "qp_full_scene.png",
                         n_mesh: int = 40) -> None:
    path = np.array(path)
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # --- feasible simplex --------------------------------------------------
    V = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    ax.add_collection3d(
        Poly3DCollection([V], facecolor='lightblue', edgecolor='navy',
                         alpha=0.30, linewidth=2))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ax.plot([V[i, 0], V[j, 0]],
                [V[i, 1], V[j, 1]],
                [V[i, 2], V[j, 2]],
                color='navy', linewidth=3)

    ax.scatter(V[:, 0], V[:, 1], V[:, 2], color='blue',
               s=100, label='feasible vertices', alpha=0.8)

    # --- objective level-sets (concentric spheres) --------------------------
    centre = np.array([0, 0, -1])
    f_vals = [f(xk, eval_hessian=False)[0] for xk in path]
    vmin, vmax = min(f_vals), max(f_vals)
    levels = np.linspace(vmin*0.8, vmax*1.2, 6)

    u, v = np.mgrid[0:np.pi:n_mesh*1j, 0:2*np.pi:n_mesh*1j]
    colours = plt.cm.viridis(np.linspace(0.2, 0.8, len(levels)))

    for col, lvl in zip(colours, levels):
        r = np.sqrt(lvl)
        xs = centre[0] + r * np.sin(u) * np.cos(v)
        ys = centre[1] + r * np.sin(u) * np.sin(v)
        zs = centre[2] + r * np.cos(u)

        # show only the part of the sphere that sits near the simplex
        mask = (xs < -0.2) | (ys < -0.2) | (zs < -0.2) | \
               (xs > 1.2) | (ys > 1.2) | (zs > 1.2)
        xsm, ysm, zsm = xs.copy(), ys.copy(), zs.copy()
        xsm[mask] = np.nan
        ysm[mask] = np.nan
        zsm[mask] = np.nan
        ax.plot_surface(xsm, ysm, zsm,
                        color=col, alpha=0.07, linewidth=0)

    # --- central-path trajectory -------------------------------------------
    for i in range(len(path)-1):
        ax.plot(path[i:i+2, 0],
                path[i:i+2, 1],
                path[i:i+2, 2],
                color=plt.cm.Reds(0.5 + 0.5*i/(len(path)-1)),
                linewidth=4)

    ax.scatter(path[:, 0], path[:, 1], path[:, 2],
               c=range(len(path)), cmap='Reds',
               s=60, edgecolors='darkred', linewidth=1)
    for k, xk in enumerate(path):
        fval = f(xk, eval_hessian=False)[0]
        ax.text(xk[0], xk[1], xk[2]+0.02, f"{fval:.3f}", fontsize=7, ha='center')

    ax.scatter(*path[0], color='green',  s=150, marker='s',
               edgecolor='darkgreen', linewidth=2, label='start point')
    ax.scatter(*path[-1], color='red',  s=150, marker='*',
               edgecolor='darkred',  linewidth=2, label='final point')

    ax.set_xlabel('$x_1$', fontweight='bold')
    ax.set_ylabel('$x_2$', fontweight='bold')
    ax.set_zlabel('$x_3$', fontweight='bold')
    ax.set_title('QP: $f(x) = x_1^2 + x_2^2 + (x_3+1)^2$\nFeasible Region: $x_1 + x_2 + x_3 = 1, x_i ≥ 0$', 
                fontsize=14, fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=35)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------
# LP  –  single 2-D scene: contours + trapezoid + central path
# -----------------------------------------------------------------
def plot_central_path_lp(path: List[np.ndarray],
                         f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
                         save_as: str = "lp_full_scene.png",
                         n_lines: int = 25) -> None:
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- feasible trapezoid -------------------------------------------------
    poly = np.array([[1, 0],
                     [2, 0],
                     [2, 1],
                     [0, 1]])            # x+y=1 gives (0,1)–(1,0) edge

    ax.fill(poly[:, 0], poly[:, 1],
            facecolor='lightgray', edgecolor='black',
            linewidth=2, alpha=0.5, label='feasible region')
    ax.scatter(poly[:, 0], poly[:, 1],
               color='black', s=90, zorder=5)

    # --- objective contours -------------------------------------------------
    x_min, x_max = -0.1, 2.2
    y_min, y_max = -0.1, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    zz = -xx - yy                       # f(x,y) = –x1 –x2

    vals = [f(xk, eval_hessian=False)[0] for xk in path]
    cmin, cmax = min(vals)-0.5, max(vals)+0.5
    levels = np.linspace(cmin, cmax, n_lines)

    cf = ax.contourf(xx, yy, zz, levels=levels,
                     cmap='coolwarm', alpha=0.6, extend='both')
    cs = ax.contour(xx, yy, zz, levels=levels[::3],
                    colors='black', linewidths=1, alpha=0.8)
    ax.clabel(cs, fmt="%.1f", fontsize=9)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('$f=-x_1-x_2$', fontweight='bold')

    # --- central-path trajectory -------------------------------------------
    for i in range(len(path)-1):
        ax.plot(path[i:i+2, 0], path[i:i+2, 1],
                color=plt.cm.Reds(0.3 + 0.7*i/(len(path)-1)),
                linewidth=4, alpha=0.9)

    ax.scatter(path[:, 0], path[:, 1],
               c=range(len(path)), cmap='Reds',
               s=80, edgecolors='darkred', linewidth=1, zorder=6)
    for k, xk in enumerate(path):
        fval = f(xk, eval_hessian=False)[0]
        ax.text(xk[0]+0.05, xk[1]+0.03, f"{fval:.3f}", fontsize=8)

    ax.scatter(*path[0],  color='green', s=200, marker='s',
               edgecolor='darkgreen', linewidth=2,  label='start point')
    ax.scatter(*path[-1], color='red',   s=200, marker='*',
               edgecolor='darkred',  linewidth=2,  label='final point')

    # arrows indicating direction
    for i in range(0, len(path)-1, max(1, len(path)//5)):
        dx, dy = path[i+1] - path[i]
        ax.arrow(path[i, 0], path[i, 1], 0.3*dx, 0.3*dy,
                 head_width=0.04, head_length=0.03,
                 fc='darkred', ec='darkred')

    ax.arrow(0.55, 0.45, 0.12, -0.12, head_width=0.04, head_length=0.03,
             fc='black', ec='black')

    ax.text(1.2, 0.6, '$x_1 + x_2 ≥ 1$', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(0.1, 0.05, '$0 ≤ x_1 ≤ 2,\\;0 ≤ x_2 ≤ 1$',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$', fontweight='bold')
    ax.set_ylabel('$x_2$', fontweight='bold')
    ax.set_title('LP  •  $f=-x_1-x_2$'
                 '\nRegion: $x_1+x_2≥1,\\;0≤x_2≤1,\\;0≤x_1≤2$',
                 fontweight='bold', pad=15)
    ax.grid(alpha=0.3); ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -------------------------------------------------------------------
# 2. OBJECTIVE-HISTORY PLOT
# -------------------------------------------------------------------
def plot_objective_history(obj_vals: List[float],
                           save_as: str = "objective_history.png",
                           title: str = "Objective vs. outer iteration") -> None:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(obj_vals)+1), obj_vals,
            marker='o', color='purple')
    ax.set_xlabel('outer iteration $k$')
    ax.set_ylabel('$f(x_k)$')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# 3. FINAL-POINT REPORT
# -------------------------------------------------------------------
def report_final(x_star: np.ndarray,
                 f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
                 ineq: List[Constraint],
                 eq_A: Optional[np.ndarray] = None,
                 eq_b: Optional[np.ndarray] = None) -> None:
    f_val, *_ = f(x_star, eval_hessian=False)
    print("\n=== FINAL CANDIDATE REPORT ===")
    print("x* =", x_star)
    print("objective =", f_val)
    for i, c in enumerate(ineq, 1):
        val, *_ = c(x_star, eval_hessian=False)
        print(f"ineq {i}:  c_i(x*) = {val}")
    if eq_A is not None:
        eq_res = eq_A @ x_star - eq_b
        print("equality residuals:", eq_res)
    print("==============================\n")