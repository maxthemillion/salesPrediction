call conda config --set ssl_verify false
call conda env create --file ./environment.yml --name salesPrediction
call conda config --set ssl_verify true
pause
