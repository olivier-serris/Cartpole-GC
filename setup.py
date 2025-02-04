from setuptools import setup, find_packages

setup(
    name="GCContinuousCartpole",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy"],
    url="http://pypi.python.org/pypi/PackageName/",
    description="evaluator system for gymnasium mazes",
    long_description=open("README.md").read(),
)
