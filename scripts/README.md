# How to run the pulsar encryption service?

## Initialization

```sh
git clone https://github.com/MarekWadinger/online_outlier_detection.git
python3 -m venv --upgrade-deps .venv
source .venv/bin/activate
cd online_outlier_detection
pip install -r requirements.txt
docker-compose up -d pulsar
```

## Run Services

Since all the services are blocking, we need to open three terminal windows with:

Terminal 1:

```sh
python scripts/pulsar_encryption_service.py
```

Terminal 2 with decryption service:

```sh
python scripts/pulsar_decryption_service.py
```

Terminal 3 with dummy producer:

```sh
python scripts/pulsar_producer.py
```

Response shall be found in Terminal 2 with

```sh
b'hello-pulsar-0'
b'hello-pulsar-1'
b'hello-pulsar-2'
```

## Clean Up

After stopping individual services stop the docker-compose service with broker

```sh
docker-compose down
```
