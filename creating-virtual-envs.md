## Option -1 Using venv 
Create your venv:
python -m venv hal_ml
Activate it:
hal_ml\Scripts\activate
Install ipykernel:
pip install ipykernel
Register it with Jupyter:
python -m ipykernel install --user --name=hal_ml --display-name "Python (HAL ML)"
Now launch Anaconda Jupyter:
jupyter notebook or jupyter lab

## Option -2 Using conda
### Since we are already using conda and it handles since it handles virtual environment + package manager + dependency manager

#### venv Manages:
Python
pip packages
Example:
python -m venv myenv

#### Conda Environment
Manages:
Python
pip packages
conda packages
different Python versions
system libraries
Kernel → Change Kernel
Now Jupyter runs using your venv packages.

### Creating conda venv
conda create -n hal_ml python=3.11
conda activate hal_ml

```bash
conda create -n hal_ml python=3.11
conda activate hal_ml

# Now try:
jupyter notebook

# You may get: 'jupyter' is not recognized...

pip install ipykernel
 # inside hal_ml.

# Then register the environment:
python -m ipykernel install --user --name hal_ml --display-name "Python (HAL ML)"

# Now launch Jupyter from Anaconda Navigator or from your base environment:
jupyter notebook

# and select:

# Kernel
  ↓
# Python (HAL ML)
```

## Suggested:

conda install numpy pandas matplotlib seaborn scikit-learn
pip install hdbscan tensorflow ipykernel

because historically:

Conda is excellent for
numpy
pandas
scikit-learn
matplotlib
scipy

These packages have compiled C/C++/Fortran code underneath.

Conda handles:

binaries
BLAS/MKL libraries
dependencies

very smoothly.

Pip is often used for
hdbscan
tensorflow
newer ML packages

because:

New versions appear on PyPI first.
Some packages are better maintained through pip.
Documentation often assumes pip.

## To be done for lstm and hdbscan
``` bash
# =============================================================================
# FLIGHT_PHASE_DETECTOR
# COMPLETE CONDA + JUPYTER SETUP
# Tested for:
#   TensorFlow 2.19
#   HDBSCAN 0.8.40
#   NumPy 1.26.4
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: OPEN ANACONDA PROMPT
# -----------------------------------------------------------------------------

# Navigate to your project folder

cd path\to\flight_phase_detector

# Example:
# cd D:\Projects\flight_phase_detector


# -----------------------------------------------------------------------------
# STEP 2: CREATE A FRESH CONDA ENVIRONMENT
# -----------------------------------------------------------------------------

conda create -n hal_fdr python=3.11 -y


# -----------------------------------------------------------------------------
# STEP 3: ACTIVATE THE ENVIRONMENT
# -----------------------------------------------------------------------------

conda activate hal_fdr


# -----------------------------------------------------------------------------
# STEP 4: UPGRADE PIP
# -----------------------------------------------------------------------------

python -m pip install --upgrade pip


# -----------------------------------------------------------------------------
# STEP 5: INSTALL ALL REQUIRED PACKAGES
# -----------------------------------------------------------------------------

pip install numpy==1.26.4
pip install pandas==2.2.3
pip install matplotlib==3.9.2
pip install seaborn==0.13.2
pip install scikit-learn==1.5.2
pip install hdbscan==0.8.40
pip install tensorflow==2.19.0
pip install jupyter==1.1.1
pip install ipykernel==6.29.5


# -----------------------------------------------------------------------------
# STEP 6: VERIFY INSTALLATION
# -----------------------------------------------------------------------------

python -c "import numpy; print('NumPy:', numpy.__version__)"

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

python -c "import hdbscan; print('HDBSCAN imported successfully')"

python -c "import sklearn; print('Scikit-Learn:', sklearn.__version__)"


# -----------------------------------------------------------------------------
# STEP 7: REGISTER ENVIRONMENT AS A JUPYTER KERNEL
# -----------------------------------------------------------------------------

python -m ipykernel install --user ^
--name hal_fdr ^
--display-name "Python (HAL FDR)"


# -----------------------------------------------------------------------------
# STEP 8: START JUPYTER NOTEBOOK
# -----------------------------------------------------------------------------

jupyter notebook


# -----------------------------------------------------------------------------
# STEP 9: INSIDE JUPYTER
# -----------------------------------------------------------------------------

# Open notebook

# Kernel
#   ↓
# Change Kernel
#   ↓
# Python (HAL FDR)

# You are now using:
#
# Python 3.11
# TensorFlow 2.19
# HDBSCAN 0.8.40
# NumPy 1.26.4


# -----------------------------------------------------------------------------
# STEP 10: TEST EVERYTHING IN A NOTEBOOK CELL
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import hdbscan
import sklearn

print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)
print("Scikit-Learn:", sklearn.__version__)
print("HDBSCAN imported successfully")


# -----------------------------------------------------------------------------
# STEP 11: SAVE REQUIREMENTS FOR GITHUB
# -----------------------------------------------------------------------------

pip freeze > requirements.txt


# -----------------------------------------------------------------------------
# STEP 12: DEACTIVATE ENVIRONMENT WHEN DONE
# -----------------------------------------------------------------------------

conda deactivate


# -----------------------------------------------------------------------------
# STEP 13: NEXT TIME YOU OPEN THE PROJECT
# -----------------------------------------------------------------------------

conda activate hal_fdr

cd path\to\flight_phase_detector

jupyter notebook
```
## More functions 
See all environments
conda env list

Example:

base
hal_ml
tensorflow_env
Switch environments
conda activate hal_ml
conda activate base
Remove environment
conda remove --name hal_ml --all
Export environment (very useful)

Once your project works:

conda env export > environment.yml

This saves all package versions.

Later:

conda env create -f environment.yml

recreates the exact environment.
