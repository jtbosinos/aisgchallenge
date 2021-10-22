# This is a sample Dockerfile
# You may adapt this to meet your environment needs

FROM python:3

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python","/main.py"]