from setuptools import find_packages, setup

setup(
    name="pixelagent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pixeltable",
    ],
    python_requires=">=3.9",
    description="A modular AI agent framework supporting OpenAI and Anthropic models",
    author="Pixeltable, Inc.",
)
