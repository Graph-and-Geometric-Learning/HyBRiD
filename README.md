# 🔥 [ICML 2024] Learning High-Order Relationships of Brain Regions

This is the official implementation of ICML 2024 paper [Learning High-Order Relationships of Brain Regions](https://arxiv.org/abs/2312.02203).


## Installation
Make sure you have [git-lfs](https://git-lfs.com/) installed in order to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the preprocessed dataset. Altenatively, you can download from [Google Drive](https://drive.google.com/drive/folders/1SvhOlPAIHVX4AYy-hU9Ik7-lKX7u1Ti2?usp=sharing).
After cloning the repo, please check the provided `environment.yml` to install the [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
```
conda env create -f environment.yml
```

### Dataset
Besides directly downloading the data from the host, `scripts/download_and_preprocess_ABIDE.py` is used to download and preprocess the ABIDE dataset.
```
python scripts/download_and_preprocess_ABIDE.py /path/to/your/output.h5
```

## Usage
### If you want to integrate HyBRiD into your own project
Copy the folder `src/hybrid` to your local storage, and use it by `from hybrid import HyBRiD`. Check the docstring and type hints in the file for more details.

**Note:**
The repo is implemented in `python3.10` and I use the new typing convention (e.g. `list[int]` instead of `List[int]`) so it is not backward compatible. However, adapting it to a lower version is always straightforward.

### If you want to run our experiment
Make sure you follow the guidance in the **Installation** section and run the following command
```shell
python main.py -c config/hybrid-piq.yaml
```
This will train the model and report the metrics on the ABIDE PIQ task.
