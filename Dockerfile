FROM python:3

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY train.csv ./

COPY hola.py ./
ENTRYPOINT ["python3","hola.py"]