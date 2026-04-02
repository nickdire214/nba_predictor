FROM apache/airflow:2.9.1
COPY requirements.txt /requirements.txt
RUN grep -iv "win32" /requirements.txt | pip install --no-cache-dir -r /dev/stdin
