FROM python:3

COPY . .

RUN pip install -r requirements.txt
EXPOSE 8899

ENTRYPOINT ["sh", "gunicorn_starter.sh"]
