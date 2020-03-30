## Installing poetry

You need to have poetry installed before starting to work on the project.

In order to install poetry simply run the following commands:

```bash
pip install --user poetry
```
> Note: If you have several python versions installed, you might want to use a specific version of Python like: `python3.7 -m pip install --user poetry`

You then need to configure poetry to create virtual environments inside your poject:

```bash
poetry config virtualenvs.in-project true
```
