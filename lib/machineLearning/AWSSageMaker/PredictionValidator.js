const cryptoWallet = require('crypto-wallet');
const Logger = cryptoWallet.Logger;
const sdkUtil = cryptoWallet.util;

class PredictionValidator {

    constructor(params = {}) {
        this.trainingData = params.trainingData;
        this.trainingDataCount = this.trainingData.target.length;
        this.numberOfPredictions = params.numberOfPredictions;
        this.predictionIndexes = [];
    }

    execute(params) {
        return this.generatePredictionIndexes()
//                .then(() => )
    }

    generatePredictionIndexes() {
        for (let index = 0; index < this.numberOfPredictions; index++) {
            const randomIndex = sdkUtil.getRandomInt(0, this.trainingDataCount);
            this.predictionIndexes.push(randomIndex);
        }
        return Promise.resolve();
    }
}

module.exports = PredictionValidator;