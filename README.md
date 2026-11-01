# On local machine
Please make sure you have python environment on your local machine.

## Install the required packages

```
pip install -r requirements.txt
```

# On Google Colab

## Upload the files in google drive

1. Please click https://workspace.google.com/products/drive/ and sign in with your google account.
2. Upload the files in google drive.
3. Open the Experiments.ipynb file in google colab.
4. Mount the google drive in google colab by running the following code:

```
from google.colab import drive
drive.mount('/content/drive')
```

5. Change the working directory to the folder where you uploaded the files by running the following code:

```
PROJECT_ROOT = "/content/drive/MyDrive/your_folder_name"  # change to your folder name
import os, sys
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

6. Install the required packages

```
!pip install -r requirements.txt
```
