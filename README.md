# PRSA

Repository contains source code to calculate phase rectified signal averaging (PRSA). PRSA is capable of detecting and quantifying quasi-periodic oscillations masked by the nonstationary nature
of composite signals and noise. More detiales could be found in:

Kantelhardt JW, Bauer A, Schumann AY, Barthel P, Schneider R, Malik M, Schmidt G. Phase-rectified signal averaging for the detection of quasi-periodicities and the prediction of cardiovascular risk. Chaos. 2007 Mar;17(1):015112. doi: 10.1063/1.2430636. 

Campana LM, Owens RL, Clifford GD, Pittman SD, Malhotra A. Phase-rectified signal averaging as a sensitive index of autonomic changes with aging. J Appl Physiol (1985). 2010 Jun;108(6):1668-73. doi: 10.1152/japplphysiol.00013.2010.


## Try Out the Code

### Install Poetry

Follow these steps to install Poetry for dependency management in Python projects.

#### Windows

1. Open PowerShell as an administrator.
2. Execute the following command:

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

#### macOS/Linux

1. Open a terminal window.
2. Run the command below:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

#### Verifying Installation

- To confirm Poetry installation, open a new terminal or command prompt and execute:

    ```bash
    poetry --version
    ```

- If Poetry is not recognized, reboot your system or manually add the Poetry bin path to your system's PATH environment variable.

#### Using Poetry

Once Poetry is installed:

1. Install project dependencies (this might take a while):

    ```bash
    poetry install --no-root
    ```

2. Activate the virtual environment:

    ```bash
    poetry shell
    ```

## Acknowledgement

Acknowledgements and credits.

## Funding

This project, "AUTOMATIC: Analysis of the relationship between the AUTOnoMic nervous system and cerebral AutoregulaTion using maChine learning approach," is financed by a grant from the SONATA-18 National Science Center (UMO-2022/47/D/ST7/00229).

## List of Research Articles

Details or references to the list of research articles related to the project.
