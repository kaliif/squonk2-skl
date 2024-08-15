# Setting up the environment
The development environment requires [Poetry](https://python-poetry.org/docs/main#installing-with-pipx).
From the project root run:


    poetry shell
    poetry install --no-root

The dependencies include torch libraries so the installation may take some time.

When done, set up pre-commit:


    pre-commit install -t commit-msg -t pre-commit
