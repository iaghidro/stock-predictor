const featureSetConfig = require('./featureSetConfig');

class TechnicalAnalysisExtractor {

    constructor(params = {}) {
        this.list = params.list;
        this.index = params.index;
    }
    
    getCurrentFeatureSet() {
        return this.list[this.index];
    }
    
    /**
     * Returns previous feature set in list. If current item is first, it returns current item.
     * @returns {Object} FeatureSet
     */
    getPreviousFeatureSet() {
        const previousIndex = this.index - 1;
        if (previousIndex < 0) {
            return this.getCurrentFeatureSet();
        }
        return this.list[previousIndex];
    }
    
    /**
     * Returns next feature set in list. If current item is last, it returns current item.
     * @returns {Object} FeatureSet
     */
    getNextFeatureSet() {
        const nextIndex = this.index + 1;
        if (nextIndex > this.list.length) {
            return this.getCurrentFeatureSet();
        }
        return this.list[nextIndex];
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
            price_close,
            BBLower,
            BBMiddle,
            BBUpper
        } = this.getCurrentFeatureSet();

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
        const featureSet = this.getCurrentFeatureSet();
        const {
            price_close
        } = featureSet;
        return (price_close < featureSet[smaName]) ? 'LOWER' : 'HIGHER';
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

}

module.exports = TechnicalAnalysisExtractor;