from setuptools import setup, find_packages

setup(
    name="data-viz-explorer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "pandas==2.0.3", 
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "scipy==1.11.2",
        "numpy==1.24.3",
        "gunicorn==21.2.0"
    ]
)