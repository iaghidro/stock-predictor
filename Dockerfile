
FROM iaghidro/machine-learning

COPY ./ /app

WORKDIR /app

RUN /app/provision.sh
