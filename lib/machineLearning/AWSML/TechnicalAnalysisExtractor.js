const _ = require('lodash');
const TDSequential = require("tdsequential");
const ExtractorBase = require('./ExtractorBase');
const featureSetConfig = require('./featureSetConfig');

class TechnicalAnalysisExtractor extends ExtractorBase {

    constructor(params = {}) {
        super(params);
        this.list = params.list;
        this.index = params.index;
    }

    getRSICategory() {
        const {
            RSI
        } = this.getCurrentFeatureSet();
        const {
            isRSIBelowThreshold,
            isRSIAboveThreshold
        } = featureSetConfig.technicalIndicators;

        if (RSI < isRSIBelowThreshold) {
            return 'LOW';
        } else if (RSI < isRSIAboveThreshold) {
            return 'MID';
        } else {
            return 'HIGH';
        }
        console.error('TechnicalAnalysisExtractor::getRSICategory ' + RSI);
        return "INVALID_RSI";
    }

    getBBCategory() {
        const {
            priceClose,
            BBLower,
            BBMiddle,
            BBUpper
        } = this.getCurrentFeatureSet();

        if (priceClose < BBLower) {
            return 'LOW_LOW';
        } else if (priceClose < BBMiddle) {
            return 'MID_LOW';
        } else if (priceClose < BBUpper) {
            return 'MID_HIGH';
        } else if (priceClose > BBUpper) {
            return 'HIGH_HIGH';
        }
    }

    getSMACategory(smaName) {
        const featureSet = this.getCurrentFeatureSet();
        const {
            priceClose
        } = featureSet;
        return (priceClose < featureSet[smaName]) ? 'LOWER' : 'HIGHER';
    }

    getMACDZeroLineCross() {
        const currentMACD = this.getCurrentFeatureSet().MACD;
        const previousMACD = this.getPreviousFeatureSet().MACD;
        if (previousMACD < 0 && currentMACD > 0) {
            return 'UP';
        }
        if (previousMACD > 0 && currentMACD < 0) {
            return 'DOWN';
        }
        return 'NONE';
    }

    getMACDSignalLineCross() {
        const currentHistogram = this.getCurrentFeatureSet().MACDHistogram;
        const previousHistogram = this.getPreviousFeatureSet().MACDHistogram;
        if (previousHistogram < 0 && currentHistogram > 0) {
            return 'UP';
        }
        if (previousHistogram > 0 && currentHistogram < 0) {
            return 'DOWN';
        }
        return 'NONE';
    }

    getTDSequential() {
        const trendLookback = 9;
        const range = super.getRange(trendLookback);
        const input = _.map(range, (featureSet) => {
            return {
                time: featureSet.timePeriodStart,
                close: featureSet.priceClose,
                high: featureSet.priceHigh,
                low: featureSet.priceLow,
                open: featureSet.priceOpen,
                volume: featureSet.volumeTraded
            };
        });
        let result = TDSequential(input);
        const lastItem = result.pop();
        if (lastItem.bearishFlip) {
            return "SELL";
        } else if (lastItem.bullishFlip) {
            return "BUY";
        }
        return "NONE";
    }

}

module.exports = TechnicalAnalysisExtractor;