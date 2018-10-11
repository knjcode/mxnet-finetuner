[MMS Arguments]
--models
{{MODEL_NAME}}=/model/{{MODEL_FILE}}

--service
optional

--gen-api
optional

--log-file
optional

--log-rotation-time
optional

--log-level
optional

--metrics-write-to
optional

[Gunicorn Arguments]

--bind
unix:/tmp/mms_app.sock

--workers
1

--worker-class
gevent

--limit-request-line
0

[Nginx Configurations]
server {
    listen       8080;

    location / {
        proxy_pass http://unix:/tmp/mms_app.sock;
    }
}

[MXNet Environment Variables]
OMP_NUM_THREADS=1
