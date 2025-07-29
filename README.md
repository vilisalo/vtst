# VTST                                                          
Python project for VTST-type kinetics code. Currently supports Gaussian 16. ORCA 6 support will be added soon.

# INSTALLATION
The package may not work out of the box in your system, depending on the native python environment and installed packages. To ensure full functionality, it is recommended to setup a suitable **miniconda** environment before running the code:

# Miniconda setup:
## Download & Install
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh 
### Setup a custom conda environment for the VTST
    conda create --name vtst
    conda install anaconda::numpy
    conda install anaconda::scipy
    conda install anaconda::tabulate
### Activating/deactivating the custom environment
    conda activate vtst
    conda deactivate vtst
# Running VTST
1. Adding the **vtst** folder into $PATH variable makes is easier to call the program from wherever.
2. The main program (**vtst.py**) has some flags in the beginning that define the QC program, whose output-files are to be read. These flags (use_gaussian or use_orca) has to be set accordingly before running the script.
3. The program can be called with or without command-line arguments. Currently the program accepts following ways of executing:
   - python vtst.py    # No command line arguments, **vtst.py** processes all suitable QC outputs with default thermochemistry parameters.
   - python vtst.py <example-file>.fchk  # **vtst.py** processes only the specified file, with default thermochemistry parameters.
   - python vtst.py <example-file>.fchk 300.0  # **vtst.py** processes only the specified file, sets the temperature to 300.0 K, while other thermochemistry parameters are default.
   - python vtst.py <example-file>.fchk 300.0 2.0  # **vtst.py** processes only the specified file, sets the temperature to 300.0 K, the pressure to 2.0 atm, while other thermochemistry parameters are default.
   - python vtst.py <example-file>.fchk 300.0 2.0 150.0  # **vtst.py** processes only the specified file, sets the temperature to 300.0 K, the pressure to 2.0 atm, and the omega_0 value of the qRRHO method to 150.0 cm-1.
4. After succesful execution, the results are printed on the screen, as well as written to an output-file called **output-vtst.out**


