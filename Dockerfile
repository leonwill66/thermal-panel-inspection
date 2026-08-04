FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
COPY webapp/requirements.txt ./webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

COPY thermal_inspector ./thermal_inspector
COPY webapp ./webapp

# webapp/data (SQLite DB + stored run images) is excluded by .dockerignore and
# must be a mounted volume at runtime - anything written to it inside the
# image without a volume mount is lost on every redeploy.
RUN mkdir -p /app/webapp/data

EXPOSE 8000

# Shell form so $PORT (injected by Fly/Railway/Render) is expanded; falls back
# to 8000 for a plain `docker run` with no PORT set.
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
