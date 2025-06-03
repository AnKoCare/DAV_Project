# Gaming Behavior Prediction Project Makefile

.PHONY: help install run dashboard notebook clean test format lint

# Default target
help:
	@echo "Gaming Behavior Prediction Project"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install project dependencies"
	@echo "  run         - Run complete analysis pipeline"
	@echo "  dashboard   - Launch interactive dashboard"
	@echo "  notebook    - Start Jupyter notebook server"
	@echo "  clean       - Clean generated files"
	@echo "  test        - Run tests (when available)"
	@echo "  format      - Format code with black"
	@echo "  lint        - Lint code with flake8"
	@echo "  setup       - Set up project environment"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Set up project environment
setup: install
	@echo "Setting up project environment..."
	python -c "from config.config import *; print('✅ Project setup complete!')"

# Run complete analysis pipeline
run:
	@echo "🎮 Running Gaming Behavior Prediction Pipeline..."
	python src/main.py

# Launch interactive dashboard
dashboard:
	@echo "🚀 Launching Gaming Behavior Analytics Dashboard..."
	python run_dashboard.py


# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	@echo "✅ Cleanup complete!"

# Run tests (placeholder for future tests)
test:
	@echo "🧪 Running tests..."
	@echo "No tests defined yet. Consider adding pytest tests!"

# Format code with black
format:
	@echo "🎨 Formatting code with black..."
	black src/ dashboard/ config/ --line-length 100
	@echo "✅ Code formatting complete!"

# Lint code with flake8
lint:
	@echo "🔍 Linting code with flake8..."
	flake8 src/ dashboard/ config/ --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting complete!"

# Quick start - install and run
quickstart: setup run
	@echo "🎉 Quick start complete! Check the reports/ directory for results."

# Demo - run dashboard with sample data
demo: setup dashboard

# Show project structure
tree:
	@echo "📁 Project Structure:"
	tree -I "__pycache__|*.pyc|*.log|.git" -L 3

# Development setup
dev-setup: install
	@echo "👨‍💻 Setting up development environment..."
	pip install black flake8 isort pytest
	@echo "✅ Development environment ready!"

# Build package
build:
	@echo "📦 Building package..."
	python setup.py sdist bdist_wheel
	@echo "✅ Package built successfully!"

# Install package in development mode
dev-install:
	@echo "🔧 Installing package in development mode..."
	pip install -e .
	@echo "✅ Package installed in development mode!" 