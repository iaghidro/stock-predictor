const _ = require('lodash');
var cryptoWallet = require('crypto-wallet');
const {
    constants,
    Slack,
    Logger,
    util
} = cryptoWallet;
const AWSSDKBase = require('./../aws/AWSSDKBase');

const appUtil = require('./../util');

class MachineLearning extends AWSSDKBase {

    constructor(params = {}) {
        super(params);
        const {bucketName, inputFileName} = params;
        this.machineLearning = this.createServiceInstance('MachineLearning');
        this.bucketName = bucketName;
        this.inputFileName = inputFileName;
    }

    train({trainingName, dataSchema, modelDataRearrangement, evaluationDataRearrangement, modelRecipe, modelParameters}) {
        this.log.info(`SageMaker::train start`);
        const generatedName = this.generateName(trainingName);

        const baseDataSourceParams = {
            dataSourceName: trainingName,
            dataSchema
        };

        return this.createDataSource(baseDataSourceParams, modelDataRearrangement)
                .then((dataSourceResponse) => {
                    this.modelDataSourceResponse = dataSourceResponse;
                })
                .then(() => this.createDataSource(baseDataSourceParams, evaluationDataRearrangement))
                .then((dataSourceResponse) => {
                    this.evaluationDataSourceResponse = dataSourceResponse;
                })
                .then(() => this.createModel(this.modelDataSourceResponse.DataSourceId, trainingName, modelRecipe, modelParameters))
                .then((modelResponse) => {
                    this.modelResponse = modelResponse;
                })
                .then(() => this.createEvaluation(this.evaluationDataSourceResponse.DataSourceId, this.modelResponse.MLModelId, trainingName))
                .then(() => util.delay(10 * 1000 * 60)) //todo: need to use aws sdk function to wait until model has finished
//                .then(() => this.createEndpoint(this.modelResponse.MLModelId))
    }

    createDataSource(params = {}, recipe) {
        this.log.info(`MachineLearning::createDataSource`);
        const dataSourceId = this.generateName(params.dataSourceName);
        let datasourceParams = {
            DataSourceId: dataSourceId, 
            DataSpec: {
                DataLocationS3: this.generateInputPath(),
                DataSchema: JSON.stringify(params.dataSchema),
//                DataSchemaLocationS3: 'STRING_VALUE'
            },
            ComputeStatistics: true,
            DataSourceName: `DATASOURCE: ${params.dataSourceName}`
        };
        if (recipe) {
            datasourceParams.DataSpec.DataRearrangement = JSON.stringify(recipe);
        }
        return new Promise((resolve, reject) => {
            this.machineLearning.createDataSourceFromS3(datasourceParams, (err, data) => {
                if (err) {
                    this.log.info(`MachineLearning::createDataSource failed`);
                    return reject(err);
                }
                this.log.info(`MachineLearning::createDataSource success`);
                resolve(data);
            });
        });
    }

    createModel(dataSourceId, modelName, modelRecipe, parameters) {
        this.log.info(`MachineLearning::createModel`);
        const modelId = this.generateName(modelName);
        var params = {
            MLModelId: modelId, 
            MLModelType: 'MULTICLASS', // todo: move into config
            TrainingDataSourceId: dataSourceId,
            MLModelName: `MODEL: ${modelName}`,
            Parameters: parameters,
            Recipe: JSON.stringify(modelRecipe)
        };
//        console.dir(modelParams);
        return appUtil.promisify(this.machineLearning, 'createMLModel', params);
    }

    createEvaluation(dataSourceId, modelId, evaluationName) {
        this.log.info(`MachineLearning::createEvaluation`);
        const evaluationId = this.generateName(evaluationName);
        var evaluationParams = {
            EvaluationDataSourceId: dataSourceId,
            EvaluationId: evaluationId,
            MLModelId: modelId,
            EvaluationName: `EVALUATION: ${evaluationName}`
        };
//        console.dir(modelParams);
        return appUtil.promisify(this.machineLearning, 'createEvaluation', evaluationParams);
    }

    createEndpoint(modelId) {
        this.log.info(`MachineLearning::createEndpoint modelId: ${modelId}`);
        const endpointParams = {
            MLModelId: modelId
        };
        return appUtil.promisify(this.machineLearning, 'createRealtimeEndpoint', endpointParams);
    }

    predict(params) {
        this.log.info(`MachineLearning::predict`);
        return this.cleanForPredict(params.Record)
                .then((cleanedFeatureSetJson) => {
                    params.Record = cleanedFeatureSetJson;
                    console.dir(params);
                    return appUtil.promisify(this.machineLearning, 'predict', params);
                });
    }

    cleanForPredict(featureSetJson) {
        this.log.info(`MachineLearning::cleanForPredict`);
        const cleanedFeatureSet = {};
        _.mapKeys(featureSetJson, (value, key) => {
            if (value === true) {
                value = 1;
            } else if (value === false) {
                value = 0;
            }
            cleanedFeatureSet[key] = value.toString();
        });
        delete cleanedFeatureSet.action;
        return Promise.resolve(cleanedFeatureSet);
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

}

module.exports = MachineLearning;
