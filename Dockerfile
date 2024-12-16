FROM python:3.10

WORKDIR /usr/app

COPY . .

RUN pip install -r requirements.docker.txt

CMD [ "python", "-u", "paper_life_simulator.py" ]
