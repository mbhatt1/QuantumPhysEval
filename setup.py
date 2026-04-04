from setuptools import find_packages, setup


setup(
    name="quantumphyseval",
    version="0.1.0",
    description="QuantumPhysEval benchmark.",
    packages=find_packages(include=["quantumphyseval", "quantumphyseval.*", "scripts"]),
    install_requires=[
        "matplotlib>=3.8",
        "numpy>=1.26",
        "openai>=1.0.0",
        "scipy>=1.12",
        "tqdm>=4.66",
    ],
    entry_points={"console_scripts": ["quantumphyseval=quantumphyseval.benchmark:main"]},
)
