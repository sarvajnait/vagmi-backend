web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --worker-tmp-dir /dev/shm main:app
