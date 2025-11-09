# Hospital Readmission Prediction Project 🏥

![img](Assets/pexels-tara-winstead-7723388.jpg)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Latest-green.svg)](https://shap.readthedocs.io/)

A comprehensive implementation of an AI Development Workflow for predicting hospital readmissions, created as part of the AI for Software Engineering course assignment.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Demo](#interactive-demo)
- [Assignment Report](#assignment-report)
- [Development](#development)
- [Contributing](#contributing)

## ✨ Features
- Synthetic patient data generation with realistic features
- Machine learning pipeline with both Logistic Regression and XGBoost
- Model interpretability using SHAP values
- Interactive Jupyter notebook demonstration
- Comprehensive unit tests
- Assignment report with full workflow documentation

## 📁 Project Structure
```
Assignment/
├── code/
│   ├── model_pipeline.py      # Main runnable script
│   ├── readmission_demo.ipynb # Interactive notebook
│   └── output/                # Generated artifacts
├── report.md                  # Assignment report
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional)
- Pandoc (optional, for PDF conversion)

### Setup Environment

1. Clone or download this repository (if using Git):
```powershell
git clone <repository-url>
cd Assignment
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## 🎮 Usage

### Run the Main Pipeline
```powershell
python .\code\model_pipeline.py
```
This will:
- Generate synthetic patient data
- Train the models
- Output evaluation metrics
- Save artifacts to `code/output/`

### Run the Interactive Notebook
```powershell
jupyter notebook .\code\readmission_demo.ipynb
```
The notebook provides:
- Step-by-step model development
- SHAP value visualizations
- Individual patient predictions
- Unit test examples

## 📊 Interactive Demo
The Jupyter notebook (`readmission_demo.ipynb`) demonstrates:
1. Data generation and preprocessing
2. XGBoost model training
3. SHAP explanations
4. Feature importance visualization
5. Individual patient analysis
6. Unit testing

## 📝 Assignment Report

The full assignment report is available in `report.md`. To convert to PDF:

```powershell
# Using Pandoc (if installed)
pandoc report.md -o report.pdf --from markdown

# Alternative: Use VS Code's PDF export
# Open report.md in VS Code
# Ctrl+Shift+P -> "Export (pdf)"
```

Report sections:
- Part 1: Short Answer Questions
- Part 2: Case Study Application
- Part 3: Critical Thinking
- Part 4: Reflection & Workflow Diagram

## 💻 Development

### Running Tests
```powershell
# From the notebook
jupyter notebook .\code\readmission_demo.ipynb
# Navigate to the Unit Tests section

# Or use pytest (if added to requirements)
pytest .\code\test_pipeline.py
```

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement
- Add cross-validation
- Implement model calibration
- Expand unit test coverage
- Add CI/CD pipeline
- Include more visualization options

## ⚠️ Disclaimer
This is an educational project. The synthetic data and models are not intended for clinical use.
