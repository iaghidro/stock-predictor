const moment = require('moment');
const _ = require('lodash');
const Big = require('big.js');
const technicalIndicators = require('technicalindicators');

class TechnicalIndicatorExtractor {

    constructor(params = {}) {
        this.list = params.list;
        this.index = params.index;
        this.technicalIndicators = technicalIndicators;
        this.roundingDecimal = 6;
        this.patternLengthTreshold = 250;
//        technicalIndicators.setConfig('precision', 6);
    }

    getRangeValues(lookbackPeriod, attributeName) {
        const range = this.index - lookbackPeriod;
        const periodStart = range >= 0 ? range : 0;
        const periodEnd = this.index + 1;
        const historicalData = this.list.slice(periodStart, periodEnd);
        return  _.map(historicalData, attributeName);
    }

    //////////////////////////
    //////// INDICATORS //////
    //////////////////////////

    calculateMovingAverage(lookbackPeriod, featureName) {
        const attributeName = featureName || 'price_close';
        const closePrices = this.getRangeValues(lookbackPeriod, attributeName);
//        console.dir(closePrices)
        const mean = _.mean(closePrices) || 0;
//        console.dir(closePrices)
        return parseFloat(Big(mean).round(this.roundingDecimal));
    }

    getTrendInput(lookbackPeriod) {
        const trendInput = {
            open: this.getRangeValues(lookbackPeriod, 'price_open'),
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            close: this.getRangeValues(lookbackPeriod, 'price_close'),
            low: this.getRangeValues(lookbackPeriod, 'price_low')
        };
        return trendInput;
    }

    getMACD(lookbackPeriod) {
        const closingPrices = this.getRangeValues(lookbackPeriod, 'price_close');
        const input = {
            values: closingPrices,
            fastPeriod: 12,
            slowPeriod: 26,
            signalPeriod: 9,
            SimpleMAOscillator: false,
            SimpleMASignal: false
        };
        const macdResponse = this.technicalIndicators.MACD.calculate(input);
        let lastItem = macdResponse.length ? macdResponse[macdResponse.length - 1] : {};
        lastItem.MACD = parseFloat(Big(lastItem.MACD || 0).round(this.roundingDecimal));
        lastItem.signal = parseFloat(Big(lastItem.signal || 0).round(this.roundingDecimal));
        lastItem.histogram = parseFloat(Big(lastItem.histogram || 0).round(this.roundingDecimal));
        return lastItem;
    }

    getRSI(lookbackPeriod, calculationPeriod) {
        const closingPrices = this.getRangeValues(lookbackPeriod, 'price_close');
        const input = {
            values: closingPrices,
            period: calculationPeriod
        };
        const response = this.technicalIndicators.RSI.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getOBV(lookbackPeriod) {
        const closingPrices = this.getRangeValues(lookbackPeriod, 'price_close');
        const volumeValues = this.getRangeValues(lookbackPeriod, 'volume_traded');
        const input = {
            close: closingPrices,
            volume: volumeValues
        };
        const response = this.technicalIndicators.OBV.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getBollingerBands(lookbackPeriod) {
        const closingPrices = this.getRangeValues(lookbackPeriod, 'price_close');
        const input = {
            period: lookbackPeriod,
            values: closingPrices,
            stdDev: 2
        };
        const response = this.technicalIndicators.BollingerBands.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : {};
        lastItem.upper = parseFloat(Big(lastItem.upper || 0).round(this.roundingDecimal));
        lastItem.lower = parseFloat(Big(lastItem.lower || 0).round(this.roundingDecimal));
        lastItem.middle = parseFloat(Big(lastItem.middle || 0).round(this.roundingDecimal));
        return lastItem;
    }

    getVWAP(lookbackPeriod) {
        const input = {
            open: this.getRangeValues(lookbackPeriod, 'price_open'),
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            close: this.getRangeValues(lookbackPeriod, 'price_close'),
            low: this.getRangeValues(lookbackPeriod, 'price_low'),
            volume: this.getRangeValues(lookbackPeriod, 'volume_traded')
        };
        const response = this.technicalIndicators.VWAP.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getADL(lookbackPeriod) {
        const input = {
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            low: this.getRangeValues(lookbackPeriod, 'price_low'),
            close: this.getRangeValues(lookbackPeriod, 'price_close'),
            volume: this.getRangeValues(lookbackPeriod, 'volume_traded')
        };
        const response = this.technicalIndicators.ADL.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getADX(lookbackPeriod, calculationPeriod) {
        const input = {
            close: this.getRangeValues(lookbackPeriod, 'price_close'),
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            low: this.getRangeValues(lookbackPeriod, 'price_low'),
            period: calculationPeriod
        };
        const response = this.technicalIndicators.ADX.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : {};
        return parseFloat(Big(lastItem.adx || 0).round(this.roundingDecimal));
    }

    getATR(lookbackPeriod, calculationPeriod) {
        const input = {
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            low: this.getRangeValues(lookbackPeriod, 'price_low'),
            close: this.getRangeValues(lookbackPeriod, 'price_close'),
            period: calculationPeriod
        };
        const response = this.technicalIndicators.ATR.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getROC(lookbackPeriod, calculationPeriod) {
        const input = {
            values: this.getRangeValues(lookbackPeriod, 'price_close'),
            period: calculationPeriod
        };
        const response = this.technicalIndicators.ROC.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    getAO(lookbackPeriod) {
        const input = {
            high: this.getRangeValues(lookbackPeriod, 'price_high'),
            low: this.getRangeValues(lookbackPeriod, 'price_low'),
            fastPeriod: 5,
            slowPeriod: 34,
            format: (a) => parseFloat(a.toFixed(2))
        };
        const response = this.technicalIndicators.AwesomeOscillator.calculate(input);
        let lastItem = response.length ? response[response.length - 1] : 0;
        return parseFloat(Big(lastItem).round(this.roundingDecimal));
    }

    //////////////////////////
    //////// PATTERNS ////////
    //////////////////////////

    getBearish(lookbackPeriod) {
        const isValidIndex = this.index >= lookbackPeriod;
        if (!isValidIndex) {
            return false;
        }
        const trendInput = this.getTrendInput(lookbackPeriod);
        return this.technicalIndicators.bearish(trendInput);
    }

    getBullish(lookbackPeriod) {
        const isValidIndex = this.index >= lookbackPeriod;
        if (!isValidIndex) {
            return false;
        }
        const trendInput = this.getTrendInput(lookbackPeriod);
        return this.technicalIndicators.bullish(trendInput);
    }

    /**
     * 
     * NOT WORKING PROPERLY
     * 
     * Available patterns:
     * isTrendingUp, 
     * isTrendingDown, 
     * hasDoubleTop, 
     * hasDoubleBottom, 
     * hasHeadAndShoulder, 
     * hasInverseHeadAndShoulder
     * @param {type} lookbackPeriod
     * @param {type} patternName
     * @returns {TechnicalIndicatorExtractor@arr;@call;technicalIndicators|Boolean}
     */
    predictPattern(lookbackPeriod, patternName) {
        const closingPrices = this.getRangeValues(lookbackPeriod, 'price_close');
        if (closingPrices.length < this.patternLengthTreshold) {
            return false;
        }
        const input = {
            values: closingPrices
        };
        return this.technicalIndicators[patternName](input);
    }

}

module.exports = TechnicalIndicatorExtractor;
