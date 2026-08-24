from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    lines = file.readlines()

requirements = [line.strip() for line in lines]
setup(name="spectra",install_requires=requirements, packages=find_packages())
