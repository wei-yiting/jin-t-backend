FROM python:3.11.14-alpine3.22

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
COPY ./app /code/app

EXPOSE 8000

# Render injects PORT=10000; DigitalOcean App Platform probes the EXPOSE
# port, so the default must stay aligned with EXPOSE above.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}