
const technicalIndicators = {
    bearishLookbackPeriod: 4,
    bullishLookbackPeriod: 4,
    MACDLookbackPeriod: 40,
    RSILookbackPeriod: 22, // should increase this to a number that all indicators use as default
    RSICalculationPeriod: 14,
    OBVLookbackPeriod: 20,
    BBLookbackPeriod: 20,
    isRSIBelowThreshold: 30,
    isRSIAboveThreshold: 70 
};

const config = {
    technicalIndicators
};

module.exports = config;