FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]