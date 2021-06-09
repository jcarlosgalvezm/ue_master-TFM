#!/bin/sh

gunicorn -b :8000 -b :8080 -t 12000 wsgi:app
