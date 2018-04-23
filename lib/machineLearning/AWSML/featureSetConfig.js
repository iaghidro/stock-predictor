
const technicalIndicators = {
    bearishLookbackPeriod: 4,
    bullishLookbackPeriod: 4,
    MACDLookbackPeriod: 40,
    RSILookbackPeriod: 22, // should increase this to a number that all indicators use as default
    RSICalculationPeriod: 14,
    OBVLookbackPeriod: 20,
    BBLookbackPeriod: 14, // try 20, trading view uses 20
    isRSIBelowThreshold: 30, //try 32
    isRSIAboveThreshold: 70 //try 68
};

const config = {
    technicalIndicators
};

module.exports = config;