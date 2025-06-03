"""
Setup script for Gaming Behavior Prediction Project
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gaming-behavior-prediction",
    version="1.0.0",
    author="Gaming Analytics Team",
    author_email="analytics@gaming.com",
    description="A comprehensive toolkit for predicting online gaming behavior and player engagement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gaming-behavior-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gaming-behavior-analysis=src.main:main",
            "gaming-behavior-dashboard=run_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gaming-behavior-prediction/issues",
        "Source": "https://github.com/yourusername/gaming-behavior-prediction",
        "Documentation": "https://gaming-behavior-prediction.readthedocs.io/",
    },
) 