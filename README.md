![build](https://github.com/jfcrenshaw/new-project/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/jfcrenshaw/new-project/branch/main/graph/badge.svg?token=3P3LYKA7EV)](https://codecov.io/gh/jfcrenshaw/new-project)

# python-package
Template repo for a new python project.

Steps:
1. Clone this repo to your local work station
2. In pyproject.toml, change name, description, homepage, and repository
3. Rename 'new_project' directory to the new package name
4. Change the docstring in new_project/\_\_init\_\_.py
5. Change all occurences of 'new_project' in .github/workflows/main.yml to the new project name
6. Go to [codecov.io](https://codecov.io/), add this repo, then click setting->badge and copy the markdown code for the badge. Paste that above.
7. Also change the build badge link above so that 'new-project' is the name of this repo
8. You can then install this package by running `poetry install` in the root directory
9. If you want to add a Jupyter kernel run the following:
```poetry run python -m ipykernel install --user --name kernel-name```

Note many of the directories in this repo are filled with place-holder files to make sure that git copies all the directories.
You may want to delete many of these and their corresponding directories.
