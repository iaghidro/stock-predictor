const _ = require('lodash');
const moment = require('moment');
const appUtil = require('./../../util');
const TechnicalIndicatorExtractor = require('./TechnicalIndicatorExtractor');
const TechnicalAnalysisExtractor = require('./TechnicalAnalysisExtractor');
const featureSetConfig = require('./featureSetConfig');

class FeatureSet {

    constructor(params = {}) {
        this.featureSetJson = params.featureSetJson;
        this.index = params.index;
        this.list = params.list;
    }

    get() {
        return this.featureSetJson;
    }

    getCleaned() {
        const {
            time_period_start,
            action,
            tenPeriodSMA,
            twentyPeriodSMA,
            thirtyPeriodSMA,
            fiftyPeriodSMA,
            hundredPeriodSMA,
            twoHundredPeriodSMA,
            isBearish,
            isBullish,
            MACD,
            RSICategory,
            BBCategory,
            isTrendingUp
            
        } = this.featureSetJson;
        return {
            time_period_start,
            action,
            tenPeriodSMA,
            twentyPeriodSMA,
            thirtyPeriodSMA,
            fiftyPeriodSMA,
            hundredPeriodSMA,
            twoHundredPeriodSMA,
            isBearish,
            isBullish,
            MACD,
            RSICategory,
            BBCategory,
            isTrendingUp
        };
    }

    validateData(timeDifferenceInMinutes) {
        if (this.index === 0) {
            return;
        }
        const previousItem = this.list[this.index - 1];
        const previousDate = moment(previousItem.time_period_start);
        const currentDate = moment(this.featureSetJson.time_period_start);

        const previousDateFormatted = previousDate.format('YYYY-MM-DDTHH:mm:ss');
        const currentDateFormatted = currentDate.format('YYYY-MM-DDTHH:mm:ss');

        if (!previousDate.isBefore(currentDate)) {
            console.error(`StockDataProcessor::validateData previous item is not before current item`);
            console.error(`StockDataProcessor::validateData previousDate: ${previousDateFormatted},current: ${currentDateFormatted}`);
        }
        const differenceInMinutes = currentDate.diff(previousDate, 'minutes');
        if (differenceInMinutes !== timeDifferenceInMinutes) {
            console.error(`StockDataProcessor::validateData invalid difference in minutes: ${differenceInMinutes}`);
            console.error(`StockDataProcessor::validateData previousDate: ${previousDateFormatted},current: ${currentDateFormatted}`);
        }
    }

    preProcessFeatures() {
        const {
            price_low,
            price_high,
            volume_traded,
            trades_count,
            price_open,
            price_close
        } = this.featureSetJson;
        this.featureSetJson = {
            time_period_start: moment(this.featureSetJson.time_period_start).unix(),
            price_low: price_low,
            price_high: price_high,
            volume_traded: volume_traded,
            trades_count: trades_count,
            price_open: price_open,
            price_close: price_close
        };
    }

    setTarget(targetLookahead, amountChangePercentageThreshold) {
        const invalidIndex = this.index > (this.list.length - 1 - targetLookahead);
        const target = 'price_close';
        if (invalidIndex) {
            return;
        }
        const currentPrice = this.featureSetJson[target];
        const periodStart = this.index + 1;
        const periodEnd = periodStart + targetLookahead + 1;
        const targetRange = this.list.slice(periodStart, periodEnd);
        const futurePrices = _.map(targetRange, target);
        const highestFuturePrice = _.max(futurePrices);
//    const lowestFuturePrice = _.min(futurePrices);
        const percentageChangeHighest = appUtil.getPercentageChange(currentPrice, highestFuturePrice);
//    const percentageChangeLowest = appUtil.getPercentageChange(currentPrice, lowestFuturePrice);
        const hasPriceIncreasedPassedThreshold = percentageChangeHighest <= (amountChangePercentageThreshold * -1);
//    const hasPriceDroppedBelowThreshold = percentageChangeLowest >= this.amountChangePercentageThreshold;
        if (hasPriceIncreasedPassedThreshold) {
            this.featureSetJson.action = "BUY";
        } else {
            this.featureSetJson.action = "HOLD";
        }
//        if (isThresholdHigherThanChange) {
//            featureSet.action = "STRONG_BUY";
//        } else if (!isThresholdHigherThanChange && percentageChange < 0) {
//            featureSet.action = "BUY";
//        } else if (isThresholdLessThanChange) {
//            featureSet.action = "STRONG_SELL";
//        } else if (!isThresholdLessThanChange && percentageChange > 0) {
//            featureSet.action = "SELL";
//        }
    }

    addTechnicalIndicators() {
        const extractor = new TechnicalIndicatorExtractor({
            list: this.list,
            index: this.index
        });
        const {price_close} = this.featureSetJson;
        const {
            bearishLookbackPeriod,
            bullishLookbackPeriod,
            MACDLookbackPeriod,
            RSILookbackPeriod,
            RSICalculationPeriod,
            OBVLookbackPeriod,
            BBLookbackPeriod,
            isRSIBelowThreshold,
            isRSIAboveThreshold
        } = featureSetConfig.technicalIndicators;
//        featureSet.trades_count = extractor.calculateMovingAverage(20, 'trades_count');
//        featureSet.price_low = extractor.calculateMovingAverage(20, 'price_low');
        this.featureSetJson.tenPeriodSMA = extractor.calculateMovingAverage(10);
        this.featureSetJson.twentyPeriodSMA = extractor.calculateMovingAverage(20);
        this.featureSetJson.thirtyPeriodSMA = extractor.calculateMovingAverage(30);
        this.featureSetJson.fiftyPeriodSMA = extractor.calculateMovingAverage(50);
        this.featureSetJson.hundredPeriodSMA = extractor.calculateMovingAverage(100);
        this.featureSetJson.twoHundredPeriodSMA = extractor.calculateMovingAverage(200);
        this.featureSetJson.isBearish = extractor.getBearish(bearishLookbackPeriod);
        this.featureSetJson.isBullish = extractor.getBullish(bullishLookbackPeriod);
        const macdResponse = extractor.getMACD(MACDLookbackPeriod);
        this.featureSetJson.MACD = macdResponse.MACD;
        this.featureSetJson.MACDSignal = macdResponse.signal;
        this.featureSetJson.RSI = extractor.getRSI(RSILookbackPeriod, RSICalculationPeriod);
        this.featureSetJson.OBV = extractor.getOBV(OBVLookbackPeriod);
        const bollingerBands = extractor.getBollingerBands(BBLookbackPeriod);
        //todo:: use bollingerBands.pb, may be percent change
        this.featureSetJson.BBUpper = bollingerBands.upper;
        this.featureSetJson.BBLower = bollingerBands.lower;
        this.featureSetJson.BBMiddle = bollingerBands.middle;
        
//        featureSet.AO = extractor.getAO(40);
//        featureSet.ROC = extractor.getROC(30,12);
//        featureSet.ADL = extractor.getADL(30);
//        featureSet.ADX = extractor.getADX(30,14);
//        featureSet.ATR = extractor.getATR(30,14);
    }
    
    addTechnicalAnalysis(){
        const extractor = new TechnicalAnalysisExtractor({
            featureSetJson: this.featureSetJson
        });
        this.featureSetJson.RSICategory = extractor.getRSICategory();
        this.featureSetJson.BBCategory = extractor.getBBCategory();
        this.featureSetJson.tenPeriodSMA = extractor.getSMACategory('tenPeriodSMA');
        this.featureSetJson.twentyPeriodSMA = extractor.getSMACategory('twentyPeriodSMA');
        this.featureSetJson.thirtyPeriodSMA = extractor.getSMACategory('thirtyPeriodSMA');
        this.featureSetJson.fiftyPeriodSMA = extractor.getSMACategory('fiftyPeriodSMA');
        this.featureSetJson.hundredPeriodSMA = extractor.getSMACategory('hundredPeriodSMA');
        this.featureSetJson.twoHundredPeriodSMA = extractor.getSMACategory('twoHundredPeriodSMA');
    }

}

module.exports = FeatureSet;