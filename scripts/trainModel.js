var app = require('../lib');
const {
    config
} = app;

const uploadDataS3BucketName = process.argv[2]; //first param

var trainModel = function () {
    console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

    const trainer = new app.ModelTrainer({
        amountChangePercentageThreshold: 0.5,
        timeDifferenceInMinutes: 30,
        targetLookahead: 3,
        recordsToRemove: 200, //todo: use percentBegin in splitting config
        recordsToRemoveTestData: 2,
        trainingName: "ethUsdCoinbase",
        bucketName: uploadDataS3BucketName,
        inputFileName: "historicalData.csv",
        propertyFilter: config.featureSet.propertyFilters.RNN,
        propertyMapping: config.featureSet.propertyMappings.coinAPI
    });
    const params = {
        limit: 15000,
        symbolId: "COINBASE_SPOT_BTC_USD",
        startTime: new Date(Date.parse("2018-08-10T00:00:00.000Z")),
        endTime: null,//new Date(Date.parse("2018-02-24T00:00:00.000Z")),
        period: "1MIN",
        fakeData: false
    };
    trainer.trainForML(params)
            .then((response) => {
                console.log(`finished training model`);
                console.dir(response);
            })
            .catch((err) => console.error(err));
};

trainModel();