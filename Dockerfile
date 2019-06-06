# start from base
FROM continuumio/miniconda3

# copy our application code
RUN mkdir /opt/app
ADD ./environment.yml /opt/app
WORKDIR /opt/app

RUN conda env create -f environment.yml
RUN  /bin/bash -c "source activate standard_env"

# expose port
EXPOSE 5000

# start app
CMD [ "python", "/opt/app/main.py" ]  