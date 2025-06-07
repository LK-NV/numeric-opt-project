import unittest
import numpy as np

from strategy_portfolio_opt.constrained_min import ConstrainedMin
from strategy_portfolio_opt.problem_formulation import f_aa, eq_constraints_mat_aa, eq_constraints_rhs_aa, ineq_aa, cov_mat

class TestMinimizer(unittest.TestCase):

    def test_aa(self):
        solver = ConstrainedMin()
        x0_aa = np.array([0.00507684, 0.01888075, 0.00028216, 0.02339314, 0.06273231, 0.01796624,
                0.00128375, 0.0286506, 0.03837643, 0.01033616, 0.05184716, 0.04885817,
                0.04430015, 0.05347783, 0.05710118, 0.00493745, 0.06086204, 0.04287345,
                0.02064393, 0.00906053, 0.03120323, 0.04730446, 0.01155924, 0.03971751,
                0.04346861, 0.0151178, 0.0055552, 0.04185903, 0.01648399, 0.02915361,
                0.01262438, 0.01934785, 0.06058349, 0.02508133])
        print(f"Initial guess for asset allocation: {x0_aa}")
        x_star_aa = solver.interior_pt(
            f=f_aa,
            ineq_constraints=ineq_aa,
            eq_constraints_mat=eq_constraints_mat_aa,
            eq_constraints_rhs=eq_constraints_rhs_aa,
            x0=x0_aa
        )
        print(f"Optimal weights for asset allocation: {x_star_aa}")

if __name__ == '__main__':
    unittest.main()

    