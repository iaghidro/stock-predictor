const _ = require('lodash');
const util = require('./../util');
const AWSSDKBase = require('./../aws/AWSSDKBase');

class SageMaker extends AWSSDKBase {

    constructor(params = {}) {
        super(params);
        const {config, bucketName, inputFileName, outputDirectoryName} = params;
        this.config = config;
        this.sageMaker = this.createServiceInstance('SageMaker');
        this.sageMakerRuntime = this.createServiceInstance('SageMakerRuntime');
        this.bucketName = bucketName;
        this.inputFileName = inputFileName;
        this.outputDirectoryName = outputDirectoryName;
    }

    train(trainingName) {
        this.log.info(`SageMaker::train start`);
        const generatedName = this.generateName(trainingName);

        const jobName = generatedName;
        const endpointConfigName = generatedName;
        const modelArtifactPath = this.generateOutputModelPath(jobName);
        const modelName = generatedName;
        const endpointName = trainingName;

        return this.createTrainingJob({jobName})
                .then((jobData) => this.waitForJobCompletion(jobData.trainingJobName))
                .then((jobData) => this.createModel(modelArtifactPath, modelName))
                .then((modelData) => this.createEndpointConfig(endpointConfigName, modelName))
                .then((endpointConfigData) => this.createEndpoint(endpointConfigName, endpointName));
    }

    createTrainingJob(params = {}) {
        this.log.info(`SageMaker::createTrainingJob start`);
        const {jobName, channelName, instanceCount, instanceType} = params;
        var params = {
            AlgorithmSpecification: {/* required */
                TrainingImage: this.config.deepARImage, /* required */
                TrainingInputMode: 'File' // Pipe | File /* required */
            },
            InputDataConfig: [/* required */
                {
                    ChannelName: channelName || this.config.channelName, /* required */
                    DataSource: {/* required */
                        S3DataSource: {/* required */
                            S3DataType: 'S3Prefix', // ManifestFile | S3Prefix /* required */
                            S3Uri: this.generateInputPath(), /* required */
                            S3DataDistributionType: 'FullyReplicated' // FullyReplicated | ShardedByS3Key
                        }
                    },
                    CompressionType: 'None', // None | Gzip
                    ContentType: 'json',
                    RecordWrapperType: 'None' // None | RecordIO
                },
                        /* more items */
            ],
            OutputDataConfig: {/* required */
                S3OutputPath: this.generateOutputPath(), /* required */
//                KmsKeyId: 'STRING_VALUE'
            },
            ResourceConfig: {/* required */
                InstanceCount: instanceCount || this.config.instanceCount, /* required */
                InstanceType: instanceType || this.config.instanceType, /* required */
                VolumeSizeInGB: this.config.volumeSize, /* required */
//                VolumeKmsKeyId: 'STRING_VALUE'
            },
            RoleArn: this.config.role, /* required */
            StoppingCondition: {/* required */
                MaxRuntimeInSeconds: this.config.maxRuntimeInSeconds
            },
            TrainingJobName: jobName, /* required MUST BE UNIQUE */
            HyperParameters: this.config.deepARHyperParams,
//            Tags: [
//                {
//                    Key: 'STRING_VALUE', /* required */
//                    Value: 'STRING_VALUE' /* required */
//                },
//            ]
        };
        return new Promise((resolve, reject) => {
            this.sageMaker.createTrainingJob(params, (err, data) => {
                if (err) {
                    this.log.info(`SageMaker::createTrainingJob failed`);
                    return reject(err);
                }
                this.log.info(`SageMaker::createTrainingJob success`);
                const response = {
                    trainingJobArn: data.TrainingJobArn,
                    trainingJobName: jobName
                };
                this.log.info(response);
                resolve(response);
            });
        });
    }

    waitForJobCompletion(trainingJobName) {
        const waitForParams = {
            TrainingJobName: trainingJobName
        };
        return super.waitFor(this.sageMaker, 'trainingJobCompletedOrStopped', waitForParams);
    }

    createModel(modelArtifactPath, modelName) {
        this.log.info(`SageMaker::createModel modelArtifactPath: ${modelArtifactPath}, modelName: ${modelName}`);
        var createModelParams = {
            ExecutionRoleArn: this.config.role, /* required */
            ModelName: modelName, /* required */
            PrimaryContainer: {/* required */
                Image: this.config.deepARImage, /* required */
//                ContainerHostname: 'STRING_VALUE',
//                Environment: {
//                    '<EnvironmentKey>': 'STRING_VALUE',
//                    /* '<EnvironmentKey>': ... */
//                },
                ModelDataUrl: modelArtifactPath
            },
//            Tags: [
//                {
//                    Key: 'STRING_VALUE', /* required */
//                    Value: 'STRING_VALUE' /* required */
//                },
//                        /* more items */
//            ]
        };
        return new Promise((resolve, reject) => {
            this.sageMaker.createModel(createModelParams, (err, data) => {
                if (err) {
                    this.log.info(`SageMaker::createModel failed`);
                    return reject(err);
                }
                this.log.info(`SageMaker::createModel success`);
                this.log.info(data);
                resolve(data);
            });
        });
    }

    createEndpointConfig(endpointConfigName, modelName) {
        this.log.info(`SageMaker::createEndpointConfig endpointConfigName: ${endpointConfigName}, modelName: ${modelName}`);
        var createEndpointConfigParams = {
            EndpointConfigName: endpointConfigName, /* required */
            ProductionVariants: [/* required */
                {
                    InitialInstanceCount: 1, /* required */
                    InstanceType: this.config.instanceType, /* required */
                    ModelName: modelName, /* required */
                    VariantName: 'defaultVariantName', /* required */
//                    InitialVariantWeight: 0.0
                },
            ],
//            KmsKeyId: 'STRING_VALUE'
        };
        return new Promise((resolve, reject) => {
            this.sageMaker.createEndpointConfig(createEndpointConfigParams, (err, data) => {
                if (err) {
                    this.log.info(`SageMaker::createEndpointConfig failed`);
                    return reject(err);
                }
                this.log.info(`SageMaker::createEndpointConfig success`);
                this.log.info(data);
                resolve(data);
            });
        });
    }

    createEndpoint(endpointConfigName, endpointName) {
        this.log.info(`SageMaker::createEndpoint endpointConfigName: ${endpointConfigName}, endpointName: ${endpointName}`);
        var createEndpointParams = {
            EndpointConfigName: endpointConfigName, /* required */
            EndpointName: endpointName, /* required */
        };
        return new Promise((resolve, reject) => {
            this.sageMaker.createEndpoint(createEndpointParams, (err, data) => {
                if (err) {
                    this.log.info(`SageMaker::createEndpoint failed`);
                    return reject(err);
                }
                this.log.info(`SageMaker::createEndpoint success`);
                this.log.info(data);
                resolve(data);
            });
        });
    }

    predict(params = {}) {
        this.log.info(`SageMaker::predict`);
        const prediction = {
            instances: params.predictions,
            configuration: {
                num_samples: this.config.numberOfSamples,
                output_types: [
                    "mean",
                    "quantiles",
//                    "samples"
                ],
                quantiles: ["0.5", "0.8"]
            }
        };
        const predictParams = {
            Body: JSON.stringify(prediction),
            EndpointName: params.endpointName, /* required */
            Accept: 'application/json',
            ContentType: 'application/json'
        };
        console.dir(predictParams)
        return new Promise((resolve, reject) => {
            this.sageMakerRuntime.invokeEndpoint(predictParams, (err, data) => {
                if (err) {
                    this.log.info(`SageMaker::predict failed`);
                    reject(err);
                } else {
                    this.log.info(`SageMaker::predict success`);
//                    data.ContentType
//                    data.InvokedProductionVariant
                    let response = data.Body.toString('utf8');
                    resolve(response);
                }
            });
        });
    }

    /////////////////////////////////
    ///////////  PRIVATE  ///////////
    /////////////////////////////////

    generateName(base) {
        const uniqueId1 = util.generateId();
        const uniqueId2 = util.generateId();
        return `${base}-${uniqueId1}-${uniqueId2}`;
    }

    generateInputPath() {
        return `s3://${this.bucketName}/${this.inputFileName}`;
    }

    generateOutputPath() {
        return `s3://${this.bucketName}/${this.outputDirectoryName}`;
    }

    generateOutputModelPath(jobName) {
        return `s3://${this.bucketName}/${this.outputDirectoryName}/${jobName}/output/model.tar.gz`;
    }

}

module.exports = SageMaker;
