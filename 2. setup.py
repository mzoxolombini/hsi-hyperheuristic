from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hsi-hyperheuristic",
    version="1.0.0",
    author="Mzoxolo Mbini",
    author_email="u16350244@tuks.co.za",
    description="Hyper-Heuristic Framework for Hyperspectral Image Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hsi-hyperheuristic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
        ],
        "gpu": [
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "hsi-hh=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.md"],
    },
)
