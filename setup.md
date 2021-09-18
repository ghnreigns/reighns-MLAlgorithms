- [Introduction](#introduction)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Setting up virtual env and requirements](#setting-up-virtual-env-and-requirements)
  - [Git:](#git)
  - [Command Line](#command-line)
  - [Documentation](#documentation)
    - [Type Hints](#type-hints)
    - [Mkdocs + Docstrings](#mkdocs--docstrings)
  - [Misc Problems](#misc-problems)

# Introduction

This guide serves as an end-to-end cycle for creating scripts for Ubuntu system.

You are assumed to have setup Git and Cuda in Ubuntu already, if not, refer to 

- [Cuda Installation](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)
- [Github Setup](https://wiki.paparazziuav.org/wiki/Github_manual_for_Ubuntu)

# Step-by-Step Guide

## Setting up virtual env and requirements

1. Open up terminal/powershell: `code path_to_script` and VScode will open up if there exists such a folder/file, if not it will create.
2. First thing you want to do is set up a virtual environtment:
    -  ```python
        sudo apt install python3.8 python3.8-venv python3-venv
        ```
    - Activate this vm by typing
        ```python
        # Create VM folder
        python3 -m venv (folder name = venv_reighns_linear_regression) # if windows, python -m venv (folder_name)
        # Activate VM folder
        source venv_reighns_linear_regression/bin/activate # if windows venv_reighns_linear_regression\Scripts\activate | if windows give you some admin bug, go to https://stackoverflow.com/questions/54776324/powershell-bug-execution-of-scripts-is-disabled-on-this-system and debug.
        # upgrade pip
        python -m pip install --upgrade pip setuptools wheel
        ```
3. Create a file named `setup.py` and `requirements.txt` concurrently. The latter should have the libraries that one is interested in having for his project while the formal is a `setup.py` file where it contains the setup object which describes how to set up our package and it's dependencies. The first several lines cover metadata (name, description, etc.) and then we define the requirements. Here we're stating that we require a Python version equal to or above 3.8 and then passing in our required packages to install_requires. Finally, we define extra requirements that different types of users may require. This is a standard practice.

    ```python
    pip install -e . -f https://download.pytorch.org/whl/torch_stable.html  # installs required packages only       -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install -e ".[dev]"                                       # installs required + dev packages
    python -m pip install -e ".[test]"                                      # installs required + test packages
    python -m pip install -e ".[docs_packages]"                             # installs required documentation packages
    ```

    > Something worth taking note is when you download PyTorch Library, there is a dependency link, you may execute as such:
    pip install -e . -f https://download.pytorch.org/whl/torch_stable.html

    - For developers, you may also need to use `test_packages, dev_packages and docs_packages` as well.
    - Entry Points: In the final lines of the `setup.py` we defined various entry points we can use to interact with the application. Here we define some console scripts (commands) we can type on our terminal to execute certain actions. For example, after we install our package, we can type the command `reighns_linear_regression ` to run the app variable inside cli.py.

## Git: 

1. `git init` to initialize the folder a local repo.
2. Create `.gitignore` to put files that you do not want `git` to commit. This is especially important for me as I often have large pdf, word and excel files.
3. `git add .` to add all existing files. To add individual file just type `git add filename.py`
4. `git status` to check status of tracked vs untracked files.
5. `git commit -a` and write your message, or `git commit -a main.py`, note if you are using vim, you need to press insert, and type your message, subsequenty press esc and type :wq to commit, or :qa to exit.
6. Note everything done above is still at local repository.
7. `git remote add origin https://github.com/reigHns/reighns-mnist.git` so that you have added this local repo to the github or bitbucket. Note to remove just use `git remote remove origin to remove`.
8. `git push origin master` to push to master branch - merge branch we do next time.
    1. One thing worth noting is that if you created the repository with some files in it, then the above steps will lead to error.
    2. Instead, to overwrite : do `git push -f origin master` . For newbies, it is safe to just init a repo with no files inside in the remote to avoid the error "Updates were rejected because the tip..."
9. Now step 8 may fail sometimes if you did not verify credentials, a fix is instead of step 7 and 8, we replace with 
    ```
    git remote add origin your-repo-http
    git remote set-url origin https://reighns@github.com/reighns/reighns-mnist.git
    git push origin master
    ```
10. `esc + q` to escape menu.

---

## Command Line

Something worth noting is we need to use dash instead of underscore when calling a function in command line.

reighns_linear_regression regression-test --solver "Batch Gradient Descent" --num-epochs 500

## Documentation

### Type Hints

### Mkdocs + Docstrings

1. Copy paste the template from Goku in, most of what he use will be in `mkdocs.yml` file. Remember to create the `mkdocs.yml` in the root path.
2. Then change accordingly the content inside `mkdocs.yml`, you can see my template that I have done up.
3. Remember to run `python -m pip install -e ".[docs_packages]" ` to make sure you do have the packages.
4. Along the way you need to create a few folders, follow the page tree in mkdocs.yml, everything should be created in `docs/` folder.
5. As an example, in our reighns-linear-regression folder, we want to show two scenarios:
    - Scenario 1: I merely want a full markdown file to show on the website. In this case, in the "watch", we specify a path we want to watch in our `docs/` folder. In this case, I created a `documentation` folder under `docs/` so I specify that. Next in the `docs/documentation/` folder I create a file called `linear_regression.md` where I dump all my markdown notes inside. Then in the `nav` tree in `mkdocs.yml`, specify
    ```
    nav:
    - Home:
        - Introduction: index.md
    - Getting started: getting_started.md
    - Detailed Notes:
        - Notes: documentation/linear_regression.md
        - Reference: documentation/reference_links.md
    ```
    Note that Home and Getting Started are optional but let us keep it for the sake of completeness. What you need to care is "Detailed Notes" and note the path I gave to them - which will point to the folders in `docs/documentation/`.

    - Scenario 2: I want a python file with detailed docstrings to appear in my static website. This is slightly more complicated. First if you want a new section of this you can create a section called `code_reference`, both under the `nav` above and also in the folder `docs/`, meaning `docs/code_reference/` must exist. Put it under watch as well. Now in `docs/code_reference/` create a file named say `linear_regression_from_scratch.md` and put `::: src.linear_regression` inside, note that you should not have space in between.
    
    
---


## Misc Problems

- How to import modules with dash

https://stackoverflow.com/questions/65861999/how-to-import-a-module-that-has-a-dash-in-its-name-without-renaming-it

- How to show nbextensions properly https://stackoverflow.com/questions/49647705/jupyter-nbextensions-does-not-appear/50663099

