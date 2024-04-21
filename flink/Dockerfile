FROM apache/flink:latest

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3 python3-pip && ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "flink.py"]