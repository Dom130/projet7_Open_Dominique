from setuptools import setup, find_packages

setup(
    name="projet7-open-dominique",
    version="1.0.0",
    packages=find_packages(include=["src*", "api*"]),
    python_requires=">=3.10",
)
