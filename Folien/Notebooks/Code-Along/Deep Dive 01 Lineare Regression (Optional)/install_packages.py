import sys
import subprocess
import os


def install_package(package_name):
    """Install a package using conda if available, otherwise pip"""

    # Check if we're in a conda environment
    in_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))

    if in_conda:
        print(f"Conda environment detected. Installing {package_name} with conda...")
        try:
            subprocess.check_call(
                ["conda", "install", "-y", package_name, "-c", "conda-forge"]
            )
        except subprocess.CalledProcessError:
            print(f"Conda install failed, trying pip...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
    else:
        print(f"Installing {package_name} with pip...", end="", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(" done.")


def install_packages(packages_to_install):
    """Install required packages for the notebook"""
    if os.path.exists(os.path.join(sys.prefix, "conda-meta")):
        print("Using conda to install packages")
    else:
        print("Using pip to install packages")

    for package in packages_to_install:
        try:
            package, package_name = package
        except ValueError:
            package_name = package

        try:
            __import__(package_name)
            print(f"Package {package} is already installed")
        except ImportError:
            install_package(package)


packages = ["ipywidgets", "plotly", ("scikit-learn", "sklearn")]
install_packages(packages)
