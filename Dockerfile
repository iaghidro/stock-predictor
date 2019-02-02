
FROM iaghidro/base:latest

COPY ./ /app

WORKDIR /app

RUN /app/provision.sh
