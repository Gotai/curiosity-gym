from setuptools import setup, find_packages

setup(name='curiosity_gym', version='0.2', packages=find_packages(include=["curiosity_gym", "curiosity_gym.*"]), install_requires=["gymnasium", "pygame"])