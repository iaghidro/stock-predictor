#!/bin/sh

# curl -X GET -H "Accept: application/json" -H "Content-Type: application/json" http://localhost:8080/ping 

curl -X POST \
  http://localhost:8080/invocations \
  -H 'accept: application/json' \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{test:"test"}'