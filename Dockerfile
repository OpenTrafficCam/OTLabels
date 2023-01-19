# ------- Development configuration
FROM python:3.10
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1=1.3.2-1 \
    gdal-bin=3.2.2+dfsg-2+deb11u2 libgdal-dev=3.2.2+dfsg-2+deb11u2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /platomo/otlabels
COPY requirements.txt requirements.txt
COPY requirements_m1.txt requirements_m1.txt
COPY requirements_dev.txt requirements_dev.txt
ENV PYTHONPATH /platomo/otlabels
RUN pip install --no-cache-dir --upgrade pip==22.3 && pip install --no-cache-dir -r requirements_dev.txt && pip install --no-cache-dir -r requirements_m1.txt
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "main.py"]
