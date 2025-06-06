## Slim Image
FROM python:3.9-slim   

## prevent python from writing pyc file
ENV PYTHONDONTWRITEBYTECODE=1 

## prevets from buffer stdout and stderr
ENV PYTHONUNBUFFERED=1 

WORKDIR /app 

# system dependencies for Streamlit and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    curl \
    libgl1 \
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt . 

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]