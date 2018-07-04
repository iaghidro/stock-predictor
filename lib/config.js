const AWSMLPropertyFilter = [
    "time_period_start",
    "action",
    "twentyPeriodSMA",
    "twoHundredPeriodSMA",
    "isBearish",
    "isBullish",
    "MACD",
    "RSICategory",
    "BBCategory",
    "TDSequential",
    "OBV",
    "RSI",
];

const RNNPropertyFilter = [
    "time_period_start",
    "price_low",
    "price_high",
    "volume_traded",
    "trades_count",
    "price_open",
    "price_close",
    "action",
    "year",
    "month",
    "dayOfWeek",
    "hour",
    "minute",
    "second"
];

const config = {
    propertyFilters: {
        AWSML: AWSMLPropertyFilter,
        RNN: RNNPropertyFilter
    }
};

module.exports = config;