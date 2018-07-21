const moment = require('moment');
const _ = require('lodash');

class StockDataProcessor {

    process(jsonData) {
        console.log(`StockDataProcessor::process`);
        return this.validateData(jsonData)
                .then(() => this.processFields(jsonData))
        // TODO: fill in missing items
    }

    validateData(jsonData) {
        console.log(`StockDataProcessor::validateData`);
        _.map(jsonData, (dataItem, index, list) => {
            if (index === 0) {
                return;
            }
            const previousItem = list[index-1];
            const previousDate = moment(previousItem.timePeriodStart);
            const currentDate = moment(dataItem.timePeriodStart);
            
            const previousDateFormatted = previousDate.format('YYYY-MM-DDTHH:mm:ss');
            const currentDateFormatted = currentDate.format('YYYY-MM-DDTHH:mm:ss');
            
            if(!previousDate.isBefore(currentDate)){
                console.error(`StockDataProcessor::validateData previous item is not before current item`);
                console.error(`StockDataProcessor::validateData previousDate: ${previousDateFormatted},current: ${currentDateFormatted}`);
            }
            const differenceInMinutes = currentDate.diff(previousDate, 'minutes');
            if(differenceInMinutes !== 1){
                console.error(`StockDataProcessor::validateData invalid difference in minutes: ${differenceInMinutes}`);
                console.error(`StockDataProcessor::validateData previousDate: ${previousDateFormatted},current: ${currentDateFormatted}`);
            }
        });
        return Promise.resolve();
    }

    processFields(jsonData) {
        console.log(`StockDataProcessor::processFielsForSageMaker`);
        const priceHighSeries = {
            start: moment(jsonData[0].timePeriodStart).format('YYYY-MM-DDTHH:mm:ss'),
            target: _.map(jsonData, (dataItem) => dataItem.priceHigh)
        };
        const cleanDataItem = [
            priceHighSeries,
        ];
        return Promise.resolve(cleanDataItem);
    }

}

module.exports = StockDataProcessor;