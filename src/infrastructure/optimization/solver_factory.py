"""
Solver Factory - Create portfolio optimizers with single line
Supports: HiGHS , CVXPY, Gurobi 
"""

from typing import List
from enum import Enum

from .base_solver import BaseSolver


class SolverType(Enum):
    """Available solver types"""
    HIGHS = "highs"
    CVXPY = "cvxpy"
    GUROBI = "gurobi"


class SolverFactory:
    """
    Factory to create portfolio optimizers.
    
    Usage:
        >>> solver = SolverFactory.create('highs')
        >>> result = solver.optimize_markowitz(returns, cov_matrix)
    
    Supported Solvers:
        - 'highs': Free, fast (via scipy)
        - 'cvxpy': Free, flexible
        - 'gurobi': Commercial, fastest (requires license)
    """
    
    @classmethod
    def create(
        cls,
        solver_type: str = "highs",
        verbose: bool = False
    ) -> BaseSolver:
        """
        Create a solver instance.
        
        Args:
            solver_type: One of 'highs', 'cvxpy', 'gurobi'
            verbose: Print solver output
        
        Returns:
            Solver instance implementing BaseSolver interface
        
        Raises:
            ValueError: If solver type unknown
            ImportError: If solver dependencies not installed
        
        Examples:
            >>> # HiGHS (recommended, free)
            >>> solver = SolverFactory.create('highs')
            
            >>> # CVXPY (alternative, free)
            >>> solver = SolverFactory.create('cvxpy')
            
            >>> # Gurobi (commercial, requires license)
            >>> solver = SolverFactory.create('gurobi')
            
            >>> # Then use it
            >>> result = solver.optimize_markowitz(returns, cov)
        """
        # Normalize solver type
        solver_type = solver_type.lower().strip()
        
        # Create appropriate solver
        if solver_type == 'highs':
            from .highs_solver import HiGHSSolver
            return HiGHSSolver(verbose=verbose)
        
        elif solver_type == 'cvxpy':
            from .cvxpy_solver import CVXPYSolver
            return CVXPYSolver(verbose=verbose)
        
        elif solver_type == 'gurobi':
            try:
                from .gurobi_solver import GurobiSolver
                return GurobiSolver(verbose=verbose)
            except ImportError:
                raise ImportError(
                    "Gurobi solver not available. You need to:\n"
                    "1. pip install gurobipy\n"
                    "2. Obtain Gurobi license (academic or commercial)\n"
                    "3. Activate license with: grbgetkey KEY\n\n"
                    "Consider using 'highs' (free) instead."
                )
        
        else:
            available = ['highs', 'cvxpy', 'gurobi']
            raise ValueError(
                f"Unknown solver '{solver_type}'. "
                f"Available solvers: {', '.join(available)}"
            )
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available solvers (with dependencies installed).
        
        Returns:
            List of solver names that can be used
        
        Example:
            >>> available = SolverFactory.list_available()
            >>> print(f"You can use: {', '.join(available)}")
            You can use: highs, cvxpy
        """
        available = []
        
        # Test each solver
        for solver_type in ['highs', 'cvxpy', 'gurobi']:
            try:
                cls.create(solver_type)
                available.append(solver_type)
            except (ImportError, ModuleNotFoundError):
                pass  # Solver not available
        
        return available
    
    @staticmethod
    def get_install_command(solver_type: str) -> str:
        """
        Get installation command for a solver.
        
        Args:
            solver_type: Solver name
        
        Returns:
            pip install command
        
        Example:
            >>> cmd = SolverFactory.get_install_command('cvxpy')
            >>> print(cmd)
            pip install cvxpy
        """
        commands = {
            'highs': 'scipy>=1.9.0  # HiGHS included',
            'cvxpy': 'cvxpy',
            'gurobi': 'gurobipy  # + license needed',
        }
        return commands.get(solver_type.lower(), 'unknown')
    
    @staticmethod
    def get_solver_info(solver_type: str) -> dict:
        """
        Get information about a solver.
        
        Args:
            solver_type: Solver name
        
        Returns:
            Dictionary with solver info
        
        Example:
            >>> info = SolverFactory.get_solver_info('highs')
            >>> print(info['description'])
            Free, fast solver via scipy
        """
        info = {
            'highs': {
                'name': 'HiGHS',
                'description': 'Free, fast solver via scipy',
                'license': 'MIT (Free)',
                'speed': 'Very Fast',
                'recommended': True,
                'install': 'pip install scipy>=1.9.0'
            },
            'cvxpy': {
                'name': 'CVXPY',
                'description': 'Flexible convex optimization',
                'license': 'Apache 2.0 (Free)',
                'speed': 'Fast',
                'recommended': False,
                'install': 'pip install cvxpy'
            },
            'gurobi': {
                'name': 'Gurobi',
                'description': 'Commercial high-performance solver',
                'license': 'Commercial (Free for academic)',
                'speed': 'Fastest',
                'recommended': False,
                'install': 'pip install gurobipy + license'
            }
        }
        return info.get(solver_type.lower(), {})


def create_optimizer(solver: str = "highs", verbose: bool = False) -> BaseSolver:
    """
    Convenience function to create optimizer.
    
    Args:
        solver: 'highs', 'cvxpy', or 'gurobi'
        verbose: Print solver output
    
    Returns:
        Optimizer instance
    
    Example:
        >>> from solver_factory import create_optimizer
        >>> optimizer = create_optimizer('highs')
        >>> result = optimizer.optimize_markowitz(returns, cov_matrix)
    """
    return SolverFactory.create(solver, verbose)


if __name__ == "__main__":
    print("="*70)
    print("SOLVER FACTORY")
    print("="*70)
    
    # List available solvers
    print("\n Checking available solvers...")
    available = SolverFactory.list_available()
    
    if available:
        print(f" Available: {', '.join(available)}")
        
        # Show info for each
        print("\n Solver Information:")
        print("-"*70)
        for solver in available:
            info = SolverFactory.get_solver_info(solver)
            print(f"\n{info['name']}:")
            print(f"  Description: {info['description']}")
            print(f"  License: {info['license']}")
            print(f"  Speed: {info['speed']}")
            print(f"  Recommended: {' Yes' if info['recommended'] else 'No'}")
    else:
        print(" No solvers available!")
        print("\nInstall at least one:")
        print("  pip install scipy>=1.9.0  # For HiGHS (recommended)")
        print("  pip install cvxpy         # For CVXPY (alternative)")
    
    print("\n" + "="*70)
    print("Usage:")
    print("  from solver_factory import create_optimizer")
    print("  optimizer = create_optimizer('highs')")
    print("  result = optimizer.optimize_markowitz(returns, cov_matrix)")
    print("="*70)
