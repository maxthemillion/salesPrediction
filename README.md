# Predict Future Sales

## How to run the prediction
This project contains a code to predict sales data.
The jupyter notebook containes data preparation. The prepared data has been saved as pickled files so that it doesn't have to be executed multiple times. The pickled data is contained in the repository as well, so if you do not change the data preparation, please only refer to the salesprediction module.

To execute the prediction itself, you need to:
1. install the python environment on your local computer
2. activate the environment and run main.py

You may either install the environment locally or host it in a docker container (and deploy it to the cloud.)

### Preferred option: Locally
1. Execute setup.bat to install the required python environment.
2. Execute start_jupyter.bat to run the notebook in your browser.
3. open the command line, activate the salesPrediciton environment and execute main.py

### Alternative option: run in docker container
Start docker and run the following commands:

> docker-compose up -d --build

> docker exec -it >>containerid<< /bin/bash

### Configuration
The module has various options for configurations. These options can be found in the config-submodule: ~/salesprediction/config/config.py. 

One of these options is to load a pretrained model and improve it by additional training. You can do so by setting  _LOAD_MODEL to True and _MODEL_NAME to the model you want to load from ~/models/. 


## How to submit a solution to kaggle
Please install the kaggle API (https://github.com/Kaggle/kaggle-api). It is included in the environment.yml as well. To use it, you have to create a kaggle account, generate a personal access token and put it into the directory specified in the documentation on github. 

Then, use the following command to submit your solution:
> kaggle competitions submit -f [filename] -m [submission message] competitive-data-science-predict-future-sales

Kaggle will then evaluate your submission.