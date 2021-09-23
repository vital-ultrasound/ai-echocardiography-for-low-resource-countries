conda update -n base -c defaults conda
conda env update --file ve.yml  --prune
#"The --prune option causes conda to remove any dependencies that are no longer required from the environment."
#https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html