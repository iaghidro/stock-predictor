//"time_period_start","price_low","price_high","volume_traded","trades_count","price_open","price_close","action","tenPeriodSMA","twentyPeriodSMA","thirtyPeriodSMA","fiftyPeriodSMA","hundredPeriodSMA","twoHundredPeriodSMA","isBearish","isBullish","MACD","MACDSignal","RSI","OBV","BBUpper","BBLower"
const base = {
    "version": "1.0",
    "rowId": "time_period_start",
    "rowWeight": null,
    "targetAttributeName": "action",
    "dataFormat": "CSV",
    "dataFileContainsHeader": true,
    "attributes": [
        {
            "attributeName": "time_period_start",
            "attributeType": "CATEGORICAL"
        },
        // candlestick start
        {
            "attributeName": "price_low",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "price_high",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "volume_traded",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "trades_count",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "price_open",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "price_close",
            "attributeType": "NUMERIC"
        },
        // candlestick end
        {
            "attributeName": "action",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "tenPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "twentyPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "thirtyPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "fiftyPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "hundredPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "twoHundredPeriodSMA",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "isBearish",
            "attributeType": "BINARY"
        },
        {
            "attributeName": "isBullish",
            "attributeType": "BINARY"
        },
//        {
//            "attributeName": "isTrendingUp",
//            "attributeType": "BINARY"
//        },
//        {
//            "attributeName": "isTrendingDown",
//            "attributeType": "BINARY"
//        },
        {
            "attributeName": "MACD",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "MACDSignal",
            "attributeType": "NUMERIC"
        },
//        {
//            "attributeName": "MACDHistogram",
//            "attributeType": "NUMERIC"
//        },
        {
            "attributeName": "RSI",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "OBV",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "BBUpper",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "BBLower",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "BBMiddle",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "isRSIAbove70",
            "attributeType": "BINARY"
        },
        {
            "attributeName": "isRSIBelow30",
            "attributeType": "BINARY"
        },
//        {
//            "attributeName": "AO",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "ROC",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "ADL",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "ADX",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "ATR",
//            "attributeType": "NUMERIC"
//        },
    ],
    "excludedAttributeNames": [
        // candlestick
        "price_low",
        "price_high",
        "price_open",
        // volume
        "volume_traded",
                // technical indicators
//        "MACD", 
//        "MACDSignal", 
//        "MACDHistogram", 
//        "RSI", 
//        "OBV",
    ]
};

const config = {
    dataSchemas: {
        base
    },
    dataRearrangement: {
        base: {
            model: {
                "splitting": {
                    "percentBegin": 0,
                    "percentEnd": 70,
                    "strategy": "random"
                }
            },
            evaluation: {
                "splitting": {
                    "percentBegin": 70,
                    "percentEnd": 100,
                    "strategy": "random"
                }
            }
        }
    },
    modelRecipes: {
        base: {
            "groups": {
                "NUMERIC_VARS_QB_500": "group('trades_count','price_close','tenPeriodSMA','twoHundredPeriodSMA','thirtyPeriodSMA','hundredPeriodSMA','fiftyPeriodSMA','twentyPeriodSMA','RSI','MACD','OBV','BBUpper','BBLower','BBMiddle')",
            },
            "assignments": {
                "normalized": "normalize(NUMERIC_VARS_QB_500)"
            },
            "outputs": [
                "ALL_BINARY",
                "ALL_CATEGORICAL",
                "quantile_bin(normalized,500)",
//                "cartesian('isBearish','isRSIAbove70')",
//                "cartesian('isBullish','isRSIBelow30')",
            ]
        }
    },
    modelParams: {
        base: {
            "sgd.maxMLModelSizeInBytes": "150554432",
            "sgd.maxPasses": "30",
            "sgd.shuffleType": "auto",
            "sgd.l2RegularizationAmount": "1.0E-04"
        }
    }
};

module.exports = config;