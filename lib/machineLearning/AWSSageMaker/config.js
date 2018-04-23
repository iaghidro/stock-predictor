const config = {
    deepARImage: '522234722520.dkr.ecr.us-east-1.amazonaws.com/forecasting-deepar:latest',
    role: 'arn:aws:iam::203258348872:role/service-role/AmazonSageMaker-ExecutionRole-20180208T131011',
    instanceCount: 1,
    channelName: 'train',
    volumeSize: 2,
    maxRuntimeInSeconds: 20 * 60,
    instanceType: 'ml.m4.xlarge',
    numberOfSamples: 10,
    // https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
    deepARHyperParams: {
        context_length: '5',
        dropout_rate: '0.05',
        epochs: '50',
        learning_rate: '0.001',
        likelihood: 'gaussian',  // deterministic-L1 or gaussian
        mini_batch_size: '32',
        num_cells: '40',
        num_layers: '3',
        prediction_length: '1', // forecast horizon
        test_quantiles: '[0.5, 0.9]',
        time_freq: 'min'
    }
};

module.exports = config;