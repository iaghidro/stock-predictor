module.exports = {
    util                    : require("./util"),
    
    AWSSDKBase              : require("./aws/AWSSDKBase"),
    MachineLearning         : require("./aws/MachineLearning"),
    S3                      : require("./aws/S3"),
    SageMaker               : require("./aws/SageMaker"),
    
    ModelTrainer            : require("./machineLearning/ModelTrainer"),
    coinAPI                 : require("./coinAPI/coinAPI"),
    MLDataProcessor         : require("./machineLearning/AWSML/StockDataProcessor"),
    SageMakerDataProcessor  : require("./machineLearning/AWSSageMaker/StockDataProcessor"),
    PredictionValidator     : require("./machineLearning/AWSSageMaker/PredictionValidator"),
};