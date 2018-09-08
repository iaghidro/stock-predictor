var app = require('../lib');
const {
    config
} = app;

const uploadDataS3BucketName = process.argv[2]; //first param

var trainModel = function () {
    console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

    const trainer = new app.ModelTrainer({
        timeDifferenceInMinutes: 30,
        bucketName: uploadDataS3BucketName,
        inputFileName: "historicalData.csv",
        coinAPIKey: "3DA8C66B-8A17-47DB-8C1D-B765C2D1C0D1"
    });
    const params = {
        limit: 50000,
        symbolId: "COINBASE_SPOT_ETH_USD",
        startTime: new Date(Date.parse("2018-07-21T01:00:00.000Z")),
        endTime: null,//new Date(Date.parse("2018-02-24T00:00:00.000Z")),
        period: "1MIN"
    };

    trainer.gatherCSVData(params)
            .then((response) => {
                console.log(`finished training model`);
                console.dir(response);
            })
            .catch((err) => console.error(err));
};

trainModel();