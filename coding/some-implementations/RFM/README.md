# Preparing

- [Anaconda](../../../README.md#python)

  - Run the following command to create a conda environment

    ```shell
    conda create -n rfm python=3.10
    ```
  - Activate the environment

    ```shell
    conda activate rfm
    ```
  - Install the necessary python packages

    ```shell
    pip install -r requirements.txt
    ```

- [Virtualenv](../../../README.md#python)

  - Run the following command to create a virtualenv environment
    ```shell
    virtualenv rfm
    ```

  - Activate the environment

    ```shell
    source rfm/bin/activate
    ```
  - Install the necessary python packages

    ```shell
    pip install -r requirements.txt
    ```
  

# Running Scripts

1. Helmholtz equation in one dimension

```shell
python locelm_helm1d_psia.py
```

2. To be added...
