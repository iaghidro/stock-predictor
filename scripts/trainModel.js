var app = require('../lib');
const {
    config
} = app;

var trainModel = function () {
    console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

    const trainer = new app.ModelTrainer({
        amountChangePercentageThreshold: 0.5,
        timeDifferenceInMinutes: 30,
        targetLookahead: 4,
        recordsToRemove: 3000,
        recordsToRemoveTestData: 20
    });
    const params = {
        limit: 15000,
        symbolId: "COINBASE_SPOT_ETH_USD",
        startTime: new Date(Date.parse("2017-08-01T00:00:00.000Z")),
        endTime: new Date(Date.parse("2018-02-24T00:00:00.000Z")),
        period: "30MIN",
        fakeData: true
    };
    trainer.trainForML(params)
            .then((response) => {
                console.log(`finished training model`);
                console.dir(response);
            })
            .catch((err) => console.error(err));
};

trainModel();