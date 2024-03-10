from setuptools import find_packages
from setuptools import setup

setup(
    name="indigo_league",
    version="0.0.1",
    url="",
    packages=find_packages(where=".", exclude=["data*", "htmlcov*", "third_party*"]),
    package_dir={"": "."},  # Specify the root directory
    license="",
    author="Alex Newgent",
    author_email="",
    description="Repository for training an AI to play pokemon",
)
