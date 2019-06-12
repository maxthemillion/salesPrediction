# Predict Future Sales

## How to run the notebook
This project contains a notebook to predict sales data. You have two options to run it. 

### Preferred option: Locally
1. Execute setup.bat to install the required python environment.
2. Execute start_jupyter.bat to run the notebook in your browser.

### Alternative option: run in docker container
Start docker and run the following commands:

> docker-compose up -d --build

> docker exec -it >>containerid<< /bin/bash

## How to submit a solution to kaggle
Please install the kaggle API (https://github.com/Kaggle/kaggle-api). It is included in the environment.yml as well. To use it, you have to create a kaggle account, generate a personal access token and put it into the directory specified in the documentation on github. 

Then, use the following command to submit your solution:
> kaggle competitions submit -f [filename] -m [submission message] competitive-data-science-predict-future-sales

Kaggle will then evaluate your submission.