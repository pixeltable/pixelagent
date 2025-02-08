from setuptools import find_packages, setup

setup(
    name="pixelagent",
    packages=find_packages(exclude=["cookbook*"]),
)
