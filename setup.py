from setuptools import setup, find_packages

setup(
    name="curiosity_gym",
    version="1.0",
    packages=find_packages(include=["curiosity_gym", "curiosity_gym.*"]),
    install_requires=["gymnasium", "pygame", "seaborn", "pandas"],
)
