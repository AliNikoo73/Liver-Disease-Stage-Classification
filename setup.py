from setuptools import setup, find_packages

setup(
    name="liver_disease_classification",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    author="AliNikoo73",
    description="A machine learning project for liver disease stage classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 