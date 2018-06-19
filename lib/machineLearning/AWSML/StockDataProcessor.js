var cryptoWallet = require('crypto-wallet');
const {
    util
} = cryptoWallet;
const moment = require('moment');
const _ = require('lodash');
const appUtil = require('./../../util');
const FeatureSet = require('./FeatureSet');

class StockDataProcessor {

    constructor(params = {}) {
        this.amountChangePercentageThreshold = params.amountChangePercentageThreshold;
        this.timeDifferenceInMinutes = params.timeDifferenceInMinutes;
        this.targetLookahead = params.targetLookAhead;
        this.calculationDelay = 5 * 1000;
        console.log(`StockDataProcessor::constructor`);
        console.dir(params);
    }

    process(jsonData) {
        console.log(`StockDataProcessor::process`);
        return this.validateData(jsonData)
                .then(() => this.preProcessFeatureSets(jsonData))
                // TODO: forward fill missing data
                .then((jsonData) => this.setTargets(jsonData))
                .then((jsonData) => this.addTechnicalIndicators(jsonData))
                .then((jsonData) => this.clean(jsonData))

    }

    preProcessFeatureSets(jsonData) {
        console.log(`StockDataProcessor::preProcessFeatureSets`);
        const featureSets = _.map(jsonData, (featureSetJson, index, list) => {
            const featureSet = new FeatureSet({
                featureSetJson,
                index,
                list
            });
            featureSet.preProcessFeatures();
            return featureSet.get();
        });
        return Promise.resolve(featureSets);
    }

    validateData(jsonData) {
        console.log(`StockDataProcessor::validateData`);
        _.map(jsonData, (featureSetJson, index, list) => {
            const featureSet = new FeatureSet({
                featureSetJson,
                index,
                list
            });
            featureSet.validateData(this.timeDifferenceInMinutes);
        });
        return Promise.resolve();
    }

    setTargets(jsonData) {
        console.log(`StockDataProcessor::setTargets`);
        const featureSets = _.map(jsonData, (featureSetJson, index, list) => {
            const featureSet = new FeatureSet({
                featureSetJson,
                index,
                list
            });
            featureSet.setTarget(this.targetLookahead, this.amountChangePercentageThreshold);
            return featureSet.get();
        });
        return Promise.resolve(featureSets);
    }

    addTechnicalIndicators(jsonData) {
        console.log(`StockDataProcessor::_addTechnicalIndicators`);
        const modifiedJson = _.map(jsonData, (featureSetJson, index, list) => {
            const featureSet = new FeatureSet({
                featureSetJson,
                index,
                list
            });
            featureSet.addTechnicalIndicators();
            featureSet.addTechnicalAnalysis();
            return featureSet.get();
        });
        return util.delay(this.calculationDelay) // may not need this to delay to give time for calculations
                .then(() => Promise.resolve(modifiedJson));
    }

    clean(jsonData) {
        console.log(`StockDataProcessor::clean`);
        const modifiedJson = _.map(jsonData, (featureSetJson, index, list) => {
            const featureSet = new FeatureSet({
                featureSetJson,
                index,
                list
            });
            return featureSet.getCleaned();
        });
        return Promise.resolve(modifiedJson)
    }

}

module.exports = StockDataProcessor;

//"price_open":0.05349299,
//"price_high":0.05350005,
//"price_low":0.05338001,
//"price_close":0.05349994,
//"volume_traded":64.90271726,
//"trades_count":37
