"# MAAE4904-Machine-Learning-Project-Fall-2024"
This is the read me file for the Machine Learning course project GitHub, created by Jack Wooldridge.

See conda_environment_setup.txt for the Conda command line input to create an environment that is compatible across the project. Note that the installation may take a few moments due to the size of some libraries. It is recommended to have 8 GB free for the libraries. Some libraries, like pytorch or opencv, may be uninstalled since they are not relevant for this project or other group members.

It is recommended to download the batch files from the cifar-10 website and note the folder they are stored in. It is also recommended to create a folder dedicated to the training and testing datasets, and for subfolders that will contain model files, plots, and metrics CSVs.

The order to run the Python files is:
    1. data_augmentation.py
    2. run_semisupervised.py
This will generate the augmented training dataset and perform semi-supervised learning to propagate the labels with a limited sample selection. After these steps, you can perform supervised learning and train a model by running either:
    - run_logistic_regression.py
    - run_random_forest.py
    - run_svm.py
    - run_voting_ensemble.py
These files import and call functions from preprocess.py and evaluate.py.

