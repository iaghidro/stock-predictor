#!/bin/sh

curl -X POST \
  http://localhost:8080/invocations \
  -H 'accept: application/json' \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{test:"test"}'