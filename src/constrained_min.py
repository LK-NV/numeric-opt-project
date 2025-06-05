import numpy as np
from typing import Callable, Optional, Tuple, TypeAlias

# f_val, grad, hess
FunctionEvaluationResult: TypeAlias = Tuple[float, np.ndarray, Optional[np.ndarray]] # type: ignore
# x, f_val, converged
OptimizationResult: TypeAlias = Tuple[np.ndarray, float, bool] # type: ignore

class ConstrainedMin:
    def __init__(self):
        """
        Initialize the constrained minimization problem.

        :param objective_function: Function to minimize.
        :param constraints: List of constraint functions.
        """
        self.opt_path: list[np.ndarray] = []
        self.obj_val_path: list[float] = []
    
    def __find_step_length(
        self,
        f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
        x: np.ndarray,
        p_k: np.ndarray,
        ineq_constraints: list[Callable[[np.ndarray, bool], FunctionEvaluationResult]],
        c1: float = 0.01,
        backtrack_factor: float = 0.5,
    ) -> float:
        """
        Back tracking line search that enforces the  Wolfe condition.

        Parameters
        ----------
        f : callable
            Objective function returning (f, g, H) where g is the gradient.
        x : np.ndarray
            Current iterate.
        p_k : np.ndarray
            Search direction (assumed descent: gᵀp_k < 0).
        ineq_constraints : list[callable]
            List of inequality constraint functions, each returning (value, gradient, hessian).
        c1 : float, optional
            Armijo constant (0 < c1 < 1). Default is 0.01.
        backtrack_factor : float, optional
            Factor by which to shrink alpha when conditions fail. Default is 0.5.

        Returns
        -------
        float
            Step length alpha satisfying the Wolfe conditions.
        """
        alpha = 1.0
        for c in ineq_constraints:
            v, g, _ = c(x, eval_hessian=False)       # v   < 0  (strictly inside)
            d = g @ p_k                              # directional derivative
            if d > 0:                                # we are moving *toward* the boundary
                alpha = min(alpha, -0.99 * v / d)    # positive step that keeps c_i < 0
        
        f0, g0, _ = f(x, eval_hessian=False)
        while True:
            new_x = x + alpha * p_k
            new_f, _, _ = f(new_x, eval_hessian=False)

            # Armijo (or Wolfe) test
            if new_f <= f0 + c1 * alpha * (g0 @ p_k):
                break
            alpha *= backtrack_factor           # shrink and retry
        return alpha

    def _make_barrier(self, f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
                     ineq_constraints: list[Callable[[np.ndarray, bool], FunctionEvaluationResult]],
                     t: float = 1.0) -> Callable[[np.ndarray, bool], FunctionEvaluationResult]:
        """
        Create a barrier function for the given objective and inequality constraints.
        :param f: Objective function to minimize.
        :param ineq_constraints: List of inequality constraint functions.
        :param t: Barrier parameter.
        :return: A function that evaluates the barrier problem.
        """
        m = len(ineq_constraints)

        def F(x: np.ndarray, *, eval_hessian: bool = True
                ) -> FunctionEvaluationResult:
            # --- objective part ---
            f_val, g, H = f(x, eval_hessian=eval_hessian)

            # --- barrier part ---
            c_vals, c_grads, c_hess = [], [], []
            for g_i in ineq_constraints:
                v, grad, hess = g_i(x, eval_hessian=eval_hessian)
                c_vals.append(v);  c_grads.append(grad);  c_hess.append(hess)

            phi = -sum(np.log(-v)          for v in c_vals)
            gphi = sum((-1/v)  * grad       for v, grad in zip(c_vals, c_grads))

            if eval_hessian:
                Hphi = (sum((1/v**2) * np.outer(grad, grad)
                            for v, grad in zip(c_vals, c_grads))
                        + sum((-1/v) * h for v, h in zip(c_vals, c_hess)))
                return t*f_val + phi, t*g + gphi, t*H + Hphi

            return t*f_val + phi, t*g + gphi, None
        return F

    def _eq_newton_step(self, g: np.ndarray, H: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the Newton step for a constrained optimization problem with equality constraints.
        :param g: Gradient vector.
        :param H: Hessian matrix.
        :param A: Matrix of equality constraints (optional).
        :return: Newton step vector.
        """
        if A is None:                     # no equalities → unconstrained Newton
            return -np.linalg.solve(H, g)

        m_eq = A.shape[0]
        KKT = np.block([[H, A.T],
                        [A, np.zeros((m_eq, m_eq))]])
        rhs = -np.concatenate([g, np.zeros(m_eq)])
        step = np.linalg.solve(KKT, rhs)
        return step[:g.shape[0]]          # extract p_k from the KKT system

    def interior_pt(
            self,
            x0: np.ndarray,
            f: Callable[[np.ndarray, bool], FunctionEvaluationResult],
            ineq_constraints: list[Callable[[np.ndarray, bool], FunctionEvaluationResult]],
            eq_constraints_mat: Optional[np.ndarray] = None,
            eq_constraints_rhs: Optional[np.ndarray] = None,
            mu: float = 10.0,
            eps: float = 1e-6,
            obj_tol: float = 1e-12,
            param_tol: float = 1e-8,
            max_newton_iter: int = 100) -> np.ndarray:
        """
        Solve a constrained minimization problem using the interior point method.
        :param x0: Initial guess for the optimization variable.
        :param f: Objective function to minimize.
        :param ineq_constraints: List of inequality constraint functions.
        :param eq_constraints_mat: Matrix of equality constraints (optional).
        :param eq_constraints_rhs: Right-hand side of equality constraints (optional).
        :param mu: Barrier parameter update factor.
        :param eps: Stopping criterion for the barrier method.
        :param obj_tol: Tolerance for the objective function value.
        :param param_tol: Tolerance for the parameters.
        :param max_newton_iter: Maximum number of inner Newton iterations.
        :return: The optimal solution.
        """
        x = x0.copy()
        t = 1.0
        m = len(ineq_constraints)

        outer_iter = 0

        while m / t > eps:
            # ---- build barrier for current t ----
            F = self._make_barrier(f, ineq_constraints, t)

            # ---- inner centering using Newton ----
            for _ in range(max_newton_iter):
                f_val, g, H = F(x, eval_hessian=True)

                p_k = self._eq_newton_step(g, H, eq_constraints_mat)

                alpha = self.__find_step_length(F, x, p_k, ineq_constraints)
                x_new = x + alpha * p_k
                f_val_new, _, _ = F(x_new, eval_hessian=False)

                # Check parameter change for convergence
                if np.linalg.norm(x_new - x) < param_tol:
                    x = x_new
                    break
                # Check objective function value for convergence
                if np.abs(f_val - f_val_new) < obj_tol:
                    x = x_new
                    break
                x = x_new

            # ---- bookkeeping for plots ----
            self.opt_path.append(x.copy())
            self.obj_val_path.append(f(x, eval_hessian=False)[0])

            # ---- next barrier parameter ----
            t *= mu
            outer_iter += 1
            print(f"[outer {outer_iter}] t={t:.1e}, f={self.obj_val_path[-1]:.6g}")

        return x
