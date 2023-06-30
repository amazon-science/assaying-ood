from setuptools import setup, find_packages

setup(
    name="Inspector",
    version="0.1",
    packages=find_packages(where="src", include=["ood_inspector", "ood_inspector.*"]),
    package_dir={"": "src"},
    scripts=["bin/run"],
)
