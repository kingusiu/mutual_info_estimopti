from setuptools import setup, find_packages

setup(
    name="mi-estimopt",
    version="0.1.0",
    description="Mutual information estimation and optimization tools",
    author="Kinga Anna Wozniak",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.7",
)
