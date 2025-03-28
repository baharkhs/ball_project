import pathlib
from setuptools import setup, find_packages

def get_requirements():
    """Read requirements.txt and return a list of dependencies."""
    here = pathlib.Path(__file__).parent.resolve()
    requirements = []
    requirements_file = here / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, encoding="utf-8") as fileh:
            for line in fileh:
                package = line.strip()
                if package and not package.startswith("#"):
                    requirements.append(package)
    return requirements

setup(
    name="ball_project",
    version="0.1.0",
    description="A simulation of bouncing balls in a cylindrical well",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics Simulation",
    ],
    keywords="molecular dynamics, ball simulation, physics",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=get_requirements(),
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False,
)
