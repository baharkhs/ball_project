from setuptools import setup, find_packages

# Read requirements from a file
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ball_project",  # Name of your package
    version="0.1.0",      # Initial version number
    #author=""
    #author_email=""
    description="A simulation of bouncing balls in a cylindrical well",
    url="https://github.com/baharkkhs/ball_project",
    packages=find_packages(),
    install_requires=required,  # Dependencies listed in requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    entry_points={
        "console_scripts": [
            "run_example=examples.run_example:example_simulation",  # Allows running "run_example" from command line
        ],
    },
)
