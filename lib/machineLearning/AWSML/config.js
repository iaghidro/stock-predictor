
const base = {
    "version": "1.0",
    "rowId": "timePeriodStart",
    "rowWeight": null,
    "targetAttributeName": "action",
    "dataFormat": "CSV",
    "dataFileContainsHeader": true,
    "attributes": [
        {
            "attributeName": "timePeriodStart",
            "attributeType": "CATEGORICAL"
        },
        // candlestick start
//        {
//            "attributeName": "priceLow",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "priceHigh",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "volumeTraded",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "tradesCount",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "priceOpen",
//            "attributeType": "NUMERIC"
//        },
//        {
//            "attributeName": "priceClose",
//            "attributeType": "NUMERIC"
//        },
        // candlestick end
        {
            "attributeName": "action",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "twentyPeriodSMA",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "twoHundredPeriodSMA",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "isBearish",
            "attributeType": "BINARY"
        },
        {
            "attributeName": "isBullish",
            "attributeType": "BINARY"
        },
        {
            "attributeName": "MACD",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "RSICategory",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "BBCategory",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "TDSequential",
            "attributeType": "CATEGORICAL"
        },
        {
            "attributeName": "OBV",
            "attributeType": "NUMERIC"
        },
        {
            "attributeName": "RSI",
            "attributeType": "NUMERIC"
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
//        // candlestick
//        "priceLow",
//        "priceHigh",
//        "priceOpen",
//        "priceClose",
//        "volumeTraded",
//        "tradesCount",
//        // technical indicators
//        "MACDSignal", 
//        "BBUpper",
//        "BBMiddle",
//        "BBLower",
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
                    "percentEnd": 90,
                    "strategy": "sequential"
                }
            },
            evaluation: {
                "splitting": {
                    "percentBegin": 90,
                    "percentEnd": 100,
                    "strategy": "sequential"
                }
            }
        }
    },
    modelRecipes: {
        base: {
            "groups": {
                "NUMERIC_VARS_QB": "group('MACD','OBV','RSI')",
            },
            "assignments": {
                "normalized": "normalize(NUMERIC_VARS_QB)"
            },
            "outputs": [
                "ALL_BINARY",
                "ALL_CATEGORICAL",
                "quantile_bin(normalized,100)",
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