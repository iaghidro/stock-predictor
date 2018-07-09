const moment = require('moment');
var chai = require('chai');
var sinon = require('sinon');
var sinonChai = require('sinon-chai');
var sinonAsPromised = require('sinon-as-promised')(Promise);
var expect = chai.expect;
var assert = chai.assert;
chai.should();
chai.use(sinonChai);

var cryptoWallet = require('crypto-wallet');
const {
    constants,
    Slack,
    Logger,
    util
} = cryptoWallet;

var app = require('./../../../lib');

const {
    ModelTrainer
} = app;

const successfulStockDataResponse = [
    {
        "timePeriodStart": "2016-05-29T00:00:00.000Z",
        "priceHigh": 3452,
    },
    {
        "timePeriodStart": "2016-05-30T00:00:00.000Z",
        "priceHigh": 456,
    },
    {
        "timePeriodStart": "2016-05-31T00:00:00.000Z",
        "priceHigh": 346,
    }, {
        "timePeriodStart": "2016-06-01T00:00:00.000Z",
        "priceHigh": 785,
    }];

describe('lib::ModelTrainer', function () {
    var modelTrainer;
    var slack;
    var errorStub;
    var getHistoricalDataStub;
    var saveCSVStub;
    var convertToCSVStub;
    var saveJsonLineStub;
    var sageMakerTrainStub;
    var machineLearningTrainStub;
    let mlCurrencyProcessorStub;
    let saveToFileStub;
    let sandbox;

    beforeEach(function (done) {
        sandbox = sinon.sandbox.create();
        modelTrainer = new ModelTrainer({
            amountChangePercentageThreshold: 0.5,
            timeDifferenceInMinutes: 30,
            targetLookahead: 4,
            recordsToRemove: 1,
            recordsToRemoveTestData: 1
        });
        modelTrainer.isTesting = true;
        slack = sandbox.stub(Slack.prototype, 'postMessage');
        errorStub = sandbox.stub(Logger.prototype, 'error');
        getHistoricalDataStub = sandbox.stub(modelTrainer.coinApiSDK, 'ohlcv_historic_data');
        getHistoricalDataStub.resolves(successfulStockDataResponse);
        // SageMaker
        saveJsonLineStub = sandbox.stub(ModelTrainer.prototype, 'saveJsonLine');
        sageMakerTrainStub = sandbox.stub(modelTrainer.sageMaker, 'train');
        // MachineLearning
        convertToCSVStub = sandbox.stub(ModelTrainer.prototype, 'converToCSV');
        saveCSVStub = sandbox.stub(ModelTrainer.prototype, 'saveCsv');
        saveToFileStub = sandbox.stub(ModelTrainer.prototype, 'saveToFile');
        saveToFileStub.resolves();
        machineLearningTrainStub = sandbox.stub(modelTrainer.machineLearning, 'train');
        mlCurrencyProcessorStub = sandbox.stub(modelTrainer.mlStockDataProcessor, 'process');
        done();
    });

    afterEach(function (done) {
        sandbox.restore();
        done();
    });

    describe('construct', function () {
        it('should set appropriate values', function (done) {
            const stockDataProcessor = new ModelTrainer();
            assert(stockDataProcessor.coinApiSDK);
            assert(stockDataProcessor.mlStockDataProcessor);
            assert(stockDataProcessor.sageMakerStockDataProcessor);
            assert(stockDataProcessor.machineLearning);
            assert(stockDataProcessor.sageMaker);
            assert(stockDataProcessor.s3);
            done();
        });
    });


    describe('trainforML', function () {
        const expectedResult =
                `"timePeriodStart","priceHigh"
"2016-05-29T00:00:00.000Z",3452
"2016-05-30T00:00:00.000Z",456`;

        //todo: need to fake out response from data processor, so only the csv conversion and removal of last 
        //items is tested and the currency data processor is tested separately
        it('should convert to csv correctly', function (done) {
            convertToCSVStub.restore();
            saveCSVStub.resolves();
            machineLearningTrainStub.resolves();
            mlCurrencyProcessorStub.resolves(successfulStockDataResponse);
            const params = {
                limit: 800,
                startTime: new Date(Date.parse("2016-04-01T00:00:00.000Z")),
                endTime: new Date(Date.parse("2017-08-21T00:00:00.000Z"))
            };
            modelTrainer.trainForML(params)
                    .then(() => {
                        const firstParam = saveCSVStub.args[0][0];
//                        console.log('(**********')
//                        console.log(JSON.stringify(firstParam))
//                        console.log('(**********')
//                        console.log(JSON.stringify(expectedResult))
//                        console.log('(**********')
                        expect(firstParam).to.equal(expectedResult);
                        done();
                    })
                    .catch((err) => console.error(err));
        });
    });

    describe('trainForSageMaker', function () {
        const expectedResult = `{"start":"2016-05-29T00:00:00","target":[3452,456]}`;

        it('should convert to sage maker format correctly', function (done) {
            saveJsonLineStub.resolves();
            sageMakerTrainStub.resolves();
            const params = {
                limit: 800,
                startTime: new Date(Date.parse("2016-04-01T00:00:00.000Z")),
                endTime: new Date(Date.parse("2017-08-21T00:00:00.000Z"))
            };
            modelTrainer.trainForSageMaker(params)
                    .then(() => {
                        const saveJsonParams = saveJsonLineStub.args[0][0];
//                        console.log('********')
//                        console.dir(saveJsonParams)
//                        console.log('********')
                        expect(saveJsonParams).to.equal(expectedResult);
                        done();
                    })
                    .catch((err) => console.error(err));
        });
    });

});