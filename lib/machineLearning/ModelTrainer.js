const createFile = require('create-file');
const _ = require('lodash');
const json2csv = require('json2csv');
const CoinApiSDK = require('./../coinAPI/coinAPI')["default"];
const MLStockDataProcessor = require('./AWSML/StockDataProcessor');
const SageMakerStockDataProcessor = require('./AWSSageMaker/StockDataProcessor');
const S3 = require('./../aws/S3');
const SageMaker = require('./../aws/SageMaker');
const MachineLearning = require('./../aws/MachineLearning');
const mlConfig = require('./AWSML/config');
const sageMakerConfig = require('./AWSSageMaker/config');

const sageMakerInputFileName = 'historicalData.json';
const sageMakerOutputDirectoryName = 'historicalDataOutput';

const coinAPIKeys = [
    "MY_COIN_API_KEY"
];

class ModelTrainer {

    constructor(params = {}) {
        this.coinApiSDK = new CoinApiSDK(coinAPIKeys[0]);
        const {
            amountChangePercentageThreshold,
            timeDifferenceInMinutes,
            targetLookahead,
            trainingName,
            bucketName,
            inputFileName
        } = params;
        console.log(`ModelTrainer::constructor`);
        console.dir(params);
        this.trainingName = trainingName;
        this.bucketName = bucketName;
        this.inputFileName = inputFileName;
        this.recordsToRemove = params.recordsToRemove;
        this.recordsToRemoveTestData = params.recordsToRemoveTestData;
        this.mlStockDataProcessor = new MLStockDataProcessor({
            amountChangePercentageThreshold,
            timeDifferenceInMinutes,
            targetLookahead
        });
        this.mlStockDataProcessor.targetLookahead = targetLookahead; ///todo fix this issue. should not have to do this
        this.sageMakerStockDataProcessor = new SageMakerStockDataProcessor();
        this.s3 = new S3();
        const sageMakerParams = {
            config: sageMakerConfig,
            bucketName: this.bucketName,
            inputFileName: sageMakerInputFileName,
            outputDirectoryName: sageMakerOutputDirectoryName,
            region: 'us-east-1'
        };
        this.sageMaker = new SageMaker(sageMakerParams);
        const machineLearningParams = {
            bucketName: this.bucketName,
            inputFileName: this.inputFileName,
            region: 'us-east-1'
        };
        this.machineLearning = new MachineLearning(machineLearningParams);
    }

    trainForML(params) {
        console.log(`ModelTrainer::trainforML`);
        const fetchFunction = params.fakeData ? 'fetchFakeData' : 'fetchData';
        return this[fetchFunction](params)
//                .then((data) => this.saveToFile(data))
                .then((jsonData) => this.mlStockDataProcessor.process(jsonData))
                //todo: add this back
                .then((cleanJsonData) => this.removeUnused(cleanJsonData))
                .then((cleanJsonData) => this.converToCSV(cleanJsonData))
                .then((csvData) => this.saveCsv(csvData))
                .then(() => this.machineLearning.train(this.trainingName, mlConfig.dataSchemas.base))
    }

    trainForSageMaker(params) {
        console.log(`ModelTrainer::trainForSageMaker`);
        const fetchFunction = params.fakeData ? 'fetchFakeData' : 'fetchData';
        return this[fetchFunction](params)
//                .then((data) => this.saveToFile(data))
                .then((cleanJsonData) => this.removeUnused(cleanJsonData))
                .then((jsonData) => this.sageMakerStockDataProcessor.process(jsonData))
                .then((jsonData) => this.convertToJsonLine(jsonData))
                .then((jsonLineData) => this.saveJsonLine(jsonLineData))
                .then(() => this.sageMaker.train(this.trainingName))
    }

    fetchFakeData(params = {}) {
        console.log(`ModelTrainer::fetchFakeData`);
        const fakeData = require('./../../trainingData/ethUsdCoinbase');
        return Promise.resolve(fakeData);
    }

    fetchData(params = {}) {
        console.log(`ModelTrainer::fetchData`);
        const {symbolId, startTime, endTime, limit, period} = params;
        console.dir(params);
        return this.coinApiSDK.ohlcv_historic_data(symbolId, period, startTime, endTime, limit);
    }

    converToCSV(data) {
        console.log(`ModelTrainer::converToCSV`);
        let csv;
        const params = {
            data
        };
        return new Promise((resolve, reject) => {
            try {
                csv = json2csv(params);
            } catch (err) {
                return reject(err);
            }
            resolve(csv);
        });
    }

    removeUnused(data = []) {
        console.log(`ModelTrainer::removeUnused initial data size: ${data.length}, recordsToRemoveTestData: ${this.recordsToRemoveTestData}`);
        let lastIndex = data.length - 1;
        if (!this.isTesting) {
            //todo: move this to config
//            remove first set of bad data
            data = data.slice(this.recordsToRemove, lastIndex);
        }
        console.log(`ModelTrainer::removeUnused recordsToRemove data size: ${data.length}`);
        lastIndex = data.length - 1;
        const testDataStart = lastIndex - this.recordsToRemoveTestData;
        const testData = data.slice(testDataStart, lastIndex);
        console.log(`************* test data *********** testDataStart: ${testDataStart}`);
        console.dir(testData);
        console.log(`ModelTrainer::removeUnused test data size: ${testData.length}`);
        console.log('************* test data ***********');
        data = data.slice(0, testDataStart);
        console.log(`ModelTrainer::removeUnused final data size: ${data.length}`);
        return Promise.resolve(data);
    }

    //http://jsonlines.org/examples/
    convertToJsonLine(data = []) {
        console.log(`ModelTrainer::convertToJsonLine`);
        function processLine(jsonObject) {
            return JSON.stringify(jsonObject);
        }
        const stringifiedArray = _.map(data, processLine);
        const stringifiedFull = stringifiedArray.join('\n');
        return Promise.resolve(stringifiedFull);
    }

    saveCsv(data = []) {
        console.log(`ModelTrainer::saveCsv length: ${data.length}`);
        return this.s3.uploadFile({
            bucketName: this.bucketName,
            file: data,
            fileName: this.inputFileName
        });
    }

    saveToFile(data) {
        const filePath = "/app/historicalData.json";
        return new Promise((resolve, reject) => {
            createFile(filePath, JSON.stringify(data), function (err) {
                if (err) {
                    return reject(err);
                }
                resolve(data);
            });
        });
    }

    saveJson(data = []) {
        console.log(`ModelTrainer::saveJson length: ${data.length}`);
        return this.s3.uploadFile({
            bucketName: this.bucketName,
            file: JSON.stringify(data),
            fileName: sageMakerInputFileName
        });
    }

    saveJsonLine(data) {
        console.log(`ModelTrainer::saveJsonLine`);
        return this.s3.uploadFile({
            bucketName: this.bucketName,
            file: data,
            fileName: sageMakerInputFileName
        });
    }
}

module.exports = ModelTrainer;
