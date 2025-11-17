FROM python:3.12.1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

# uses exact libraries in uv.lock file
RUN uv sync --locked 

COPY predict.py model.bin ./

EXPOSE 9696

# entry point for running the app
#no need of uv run cuz there is only on env inside docker
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]