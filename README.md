# Project Title

A brief description of what this project does and who it's for.

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Create Python Environment](#create-python-environment)
    - [Install Dependencies](#install-dependencies)
    - [Git LFS](#git-lfs)
  - [Running the App](#running-the-app)

## Installation

Follow these steps to set up your environment and install the necessary dependencies.

### Create Python Environment

1. Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).

2. Create a virtual environment:

    ```sh
    python -m venv env
    ```

3. Activate the virtual environment:

    - On Windows:
    
        ```sh
        .\env\Scripts\activate
        ```


### Install Dependencies

1. Make sure you have `pip` installed. It usually comes with Python. To check if you have `pip`:

    ```sh
    pip --version
    ```

2. Install the required packages using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

### Git LFS

1. Install Git LFS from the official [Git LFS website](https://git-lfs.github.com/).

2. Initialize Git LFS in your repository:

    ```sh
    git lfs install
    ```

3. Pull the LFS files:

    ```sh
    git lfs pull
    ```

## Running the App

To run the Streamlit application:

```sh
streamlit run main.py
