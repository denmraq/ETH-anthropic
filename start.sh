#!/bin/sh
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-10000}" --workers 1
