#!/bin/bash

AWS_KEY_ID=$1;
AWS_ACCESS_KEY=$2;

echo "::start:: KEEP IN SYNC WITH HOST MACHINE'S TIME ";
/etc/init.d/ntp restart;


# CONFIGURE AWS
echo "::start:: APP_NANE: ${APP_NANE} ";
mkdir -p ~/.aws;
touch ~/.aws/config || exit;
printf '[default]\noutput = json\nregion = us-west-1' > ~/.aws/config;
touch ~/.aws/credentials || exit;
printf "[default]\naws_access_key_id = ${AWS_KEY_ID}\naws_secret_access_key = ${AWS_ACCESS_KEY}" > ~/.aws/credentials;

#START CRON IN FOREGROUND. THIS WILL KEEP CONTAINER UP AND RUN CRON SERVICES
cron -f;