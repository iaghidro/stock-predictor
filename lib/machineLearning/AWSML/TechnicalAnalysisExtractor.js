const featureSetConfig = require('./featureSetConfig');

class TechnicalAnalysisExtractor {

    constructor(params = {}) {
        this.featureSetJson = params.featureSetJson;
    }

    getRSICategory() {
        const {
            RSI
        } = this.featureSetJson;
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
            price_close,
            BBLower,
            BBMiddle,
            BBUpper
        } = this.featureSetJson;

        if (price_close < BBLower) {
            return 'LOW_LOW';
        } else if (price_close < BBMiddle) {
            return 'MID_LOW';
        } else if (price_close < BBUpper) {
            return 'MID_HIGH';
        } else if (price_close > BBUpper) {
            return 'HIGH_HIGH';
        }
    }

    getSMACategory(smaName) {
        const {
            price_close
        } = this.featureSetJson;
        return (price_close < this.featureSetJson[smaName]) ? 'LOWER' : 'HIGHER';
    }

}

module.exports = TechnicalAnalysisExtractor;