#Initial Docker Image:
docker pull agrigorev/zoomcamp-model:2025

docker build -t lead-score-conversion .
docker run -it --rm -p 9696:9696 lead-score-conversion

uv run uvicorn predict:app --host 0.0.0.0 --port 9696
