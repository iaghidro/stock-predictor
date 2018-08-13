#!/bin/bash

AWS_KEY_ID=$1;
AWS_ACCESS_KEY=$2;

echo "::start:: KEEP IN SYNC WITH HOST MACHINE'S TIME ";
/etc/init.d/ntp restart;


# CONFIGURE AWS
echo "::start:: APP_NANE: ${APP_NANE} ";
mkdir -p ~/.aws;
touch ~/.aws/config || exit;
printf '[default]\noutput = json\nregion = us-east-1' > ~/.aws/config;
touch ~/.aws/credentials || exit;
printf "[default]\naws_access_key_id = ${AWS_KEY_ID}\naws_secret_access_key = ${AWS_ACCESS_KEY}" > ~/.aws/credentials;

# INSTALL DEPENDENCIES
pip3 install pytest
pip3 install ta
cd tmp
bash /app/fastAI/installScripts/1.install-utils.sh
bash /app/fastAI/installScripts/2.install-cuda_8.sh
bash /app/fastAI/installScripts/3.install-repos.sh
bash /app/fastAI/installScripts/4.install-anaconda-env.sh
conda env update -f environment-cpu.yml

#START CRON IN FOREGROUND. THIS WILL KEEP CONTAINER UP AND RUN CRON SERVICES
cron -f;