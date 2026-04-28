from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent

setup(
    name="tcfmad",
    version="0.1.0",
    description="tcfmad",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
    ],
    packages=["tcfmad"] + find_packages(where="tcfmad"),
    package_dir={"tcfmad": "tcfmad", "": "tcfmad"},
    package_data={"": ["*.yaml", "*.yml", "*.json"]},
    entry_points={
        "console_scripts": [
            "tcfmad = tcfmad.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
