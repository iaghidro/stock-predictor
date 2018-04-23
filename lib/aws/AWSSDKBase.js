const AWS = require('aws-sdk');
const cryptoWallet = require('crypto-wallet');
const Logger = cryptoWallet.Logger;
const sdkUtil = cryptoWallet.util;

const services = [
    'EC2', 
    'S3', 
    'SageMaker', 
    'SageMakerRuntime', 
    'MachineLearning'
];

const defaultRegion = "us-west-1";

class AWSSDKBase {

    constructor(params = {}) {
        const logId = `AWSSDKBase::${this.constructor.name}`;
        this.log = params.log ? params.log : Logger.create(logId);
        AWS.config.update({
            region: params.region || defaultRegion,
            s3: '2006-03-01',
            ec2: '2016-11-15',
            sagemaker: '2017-07-24',
            sagemakerruntime: '2017-05-13',
            machinelearning: '2014-12-12'
        });
        this.log.info(`constructor`);
    }

    createServiceInstance(serviceType) {
        const validService = services.includes(serviceType);
        if (validService) {
            return new AWS[serviceType]({

            });
        } else {
            throw new Error(`createServiceInstance Invaling AWS service type: ${serviceType}`);
        }

    }

    waitFor(service, state, params = {}, inputParams) {
        this.log.info(`waitFor state: ${state}`);
        return new Promise((resolve, reject) => {
            service.waitFor(state, params, (err, data) => {
                if (err) {
                    this.log.info(`waitFor failed`);
                    reject(err);
                }
                this.log.info(`waitFor success`);
                resolve(data);
            });
        });
    }

}

module.exports = AWSSDKBase;
