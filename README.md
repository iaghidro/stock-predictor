# Stock Predictor

A NodeJS stock predictor application using AWS ML, and SageMaker. 

# Prerequisites:

* Requires NodeJS 7.7.3 +
* Must configure AWS to give access to S3, ML, and SageMaker services. This can be done manually or access keys can be passed to the start.sh script
* Must give AWS ML and Sage Maker access to the S3 upload bucket where the training data will be uploaded 


# Create/Train a machine learning model

This script does the following:
* retrieves and cleans stock data
* performs technical analysis
* feeds data into AWS ML
* generates Datasets (70% for training, 30% for evaluation)
* creates a Model
* creates an evaluation
* creates a prediction endpoint

    npm run trainModel uploadDataS3BucketName

# Make a prediction

This script makes a prediction to the prediction endpoint with the data defined in the predict script located in /scripts/predict.js

    npm run predict

# Running Tests:

1) run tests

NodeJS

    npm test

Python (cd into /lib directory then run this command)

    npm run test:p

2) grep for tests

    npm run test:grep 'MyClass'