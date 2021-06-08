#!/bin/sh

gunicorn -b 0.0.0.0:8899 -t 12000 wsgi:app
