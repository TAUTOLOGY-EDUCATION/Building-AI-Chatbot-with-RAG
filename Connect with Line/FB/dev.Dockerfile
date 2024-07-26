FROM python:3.12.4-bullseye
WORKDIR /container
EXPOSE 80


RUN pip install pdm
COPY ./container/pyproject.toml .
COPY ./container/pdm.lock .
COPY ./container/README.md .

RUN pdm install -G dev

COPY ./container .

ENV TZ=Asia/Bangkok

# CMD tail -f /dev/null
CMD pdm run uvicorn main:app --host 0.0.0.0 --port 80 --reload

