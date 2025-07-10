# Dynamic Differential Microscopy (DDM)

For a reference of the underlying theory of DDM, see the seminal textbook by [Berne & Peocora](https://www.eng.uc.edu/~beaucag/Classes/Properties/Books/Bruce%20J.%20Berne,%20Robert%20Pecora%20-%20Dynamic%20Light%20Scattering_%20With%20Applications%20to%20Chemistry,%20Biology,%20and%20Physics-John%20Wiley%20&%20Sons,%20Inc.%20(2000).pdf).

## Cloning this repository

It is recommended that you clone this repository (see [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for how to do this) to stay up to date with any changes made.

## Installing Requirements

All scripts in the `build` folder require an up to date installation of Python - if you do not have Python installed on your machine, see [here](https://www.python.org/downloads/).

To install the required packages, it is recommended to use a virtual environment (see [here](https://docs.python.org/3/library/venv.html)). This avoids any conflicts with package versions already existing on your machine's global environment. Setting up the virtual environment *should* be necessary - if you run into package related issues it may be a solution.

To install the required packages, `cd` into the `build` directory and run `python install_requirements.py` in your command terminal.

## Use

The [`example.py`](build/example.py) script is provided showing how to use the `DDM` custom class. It is important to read the implementation to follow the algorithm properly and debug/make edits.
