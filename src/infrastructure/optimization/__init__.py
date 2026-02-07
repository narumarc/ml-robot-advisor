"""
Portfolio Optimizers Package
Supports multiple solvers: HiGHS (free), CVXPY (free), Gurobi (commercial)
"""

from .base_solver import BaseSolver, OptimizationResult
from .solver_factory import SolverFactory, create_optimizer
from .highs_solver import HiGHSSolver
from .cvxpy_solver import CVXPYSolver

__all__ = [
    # Main interface
    'create_optimizer',
    'SolverFactory',
    
    # Base classes
    'BaseSolver',
    'OptimizationResult',
    
    # Concrete solvers
    'HiGHSSolver',
    'CVXPYSolver',
]

__version__ = '1.0.0'

# Quick usage guide
__doc__ = """
Portfolio Optimization with Multiple Solvers
=============================================

Quick Start:
-----------
>>> from optimizers import create_optimizer
>>> 
>>> # Create optimizer (HiGHS recommended, free)
>>> optimizer = create_optimizer('highs')
>>> 
>>> # Optimize portfolio
>>> result = optimizer.optimize_markowitz(
...     expected_returns=expected_returns,
...     cov_matrix=cov_matrix
... )
>>> 
>>> # Get results
>>> if result.success:
...     print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
...     for asset, weight in result.weights.items():
...         print(f"  {asset}: {weight*100:.1f}%")

Available Solvers:
-----------------
- 'highs': FREE, fast (via scipy) ⭐ RECOMMENDED
- 'cvxpy': FREE, flexible
- 'gurobi': Commercial, fastest (requires license)

Check available solvers:
-----------------------
>>> from optimizers import SolverFactory
>>> available = SolverFactory.list_available()
>>> print(available)
['highs', 'cvxpy']

Switch solvers easily:
---------------------
>>> # Method 1: Change one line
>>> optimizer = create_optimizer('highs')  # or 'cvxpy', 'gurobi'
>>> 
>>> # Method 2: Factory
>>> from optimizers import SolverFactory
>>> optimizer = SolverFactory.create('highs', verbose=True)

All solvers share the same interface (BaseSolver).
"""
