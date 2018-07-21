const AWSMLPropertyFilter = [
    "timePeriodStart",
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
    "timePeriodStart",
    "priceLow",
    "priceHigh",
    "volumeTraded",
    "tradesCount",
    "priceOpen",
    "priceClose",
    "action",
    "year",
    "month",
    "dayOfWeek",
    "hour",
    "minute",
    "second"
];

const config = {
    featureSet: {
        propertyFilters: {
            AWSML: AWSMLPropertyFilter,
            RNN: RNNPropertyFilter
        },
        propertyMappings: {
            coinAPI: {
                timePeriodStart : "time_period_start",
                timePeriodEnd : "time_period_end",
                priceOpen: "price_open",
                priceClose: "price_close",
                priceHigh: "price_high",
                priceLow: "price_low",
                volumeTraded: "volume_traded",
                tradesCount: "trades_count",
            }
        }
    }
};

module.exports = config;