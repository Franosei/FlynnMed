FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system flynnmed \
    && useradd --system --gid flynnmed --home-dir /app --no-create-home flynnmed

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=flynnmed:flynnmed . .
COPY --chown=flynnmed:flynnmed --from=frontend-builder /build/frontend/dist ./frontend/dist
RUN mkdir -p /app/data && chown -R flynnmed:flynnmed /app/data

EXPOSE 8000
USER flynnmed
CMD ["sh", "/app/scripts/start.sh"]
