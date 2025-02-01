from setuptools import setup, find_packages

setup(
    name="pixelagent",
    packages=find_packages(exclude=["cookbook*"]),
)
