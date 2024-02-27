# PRSA

Description of the project or document.

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
