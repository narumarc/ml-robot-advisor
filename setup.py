"""
Setup configuration for Robo-Advisor Portfolio Optimization Platform.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Robo-Advisor Portfolio Optimization Platform"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="robo-advisor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Portfolio Optimization Platform with ML and Mathematical Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/robo-advisor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.1",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.2",
        ],
        "ml": [
            "scikit-learn>=1.3.2",
            "tensorflow>=2.15.0",
            "torch>=2.1.2",
            "xgboost>=2.0.3",
            "lightgbm>=4.1.0",
        ],
        "optimization": [
            "gurobipy>=11.0.0",
            "ortools>=9.8.3296",
            "cvxpy>=1.4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "robo-advisor=src.presentation.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
