FROM python:3.8-slim

WORKDIR /model-service
COPY . .

RUN apt-get update -y
RUN apt-get install -y iputils-ping

# Disable pip cache to shrink the image size a little bit, since it does not need to be re-installed
RUN pip install -r requirements.txt --no-cache-dir

# Port will be exposed, for documentation only
EXPOSE 8081
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
