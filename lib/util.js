var async = require('async');

function doUntil(doFunction, conditionFunction) {
    var self = this;
    return new Promise((resolve, reject) => {
        async.doUntil(
                function executeFunction(eachCb) {
                    doFunction.call(self)
                            .then(eachCb)
                            .catch(eachCb);
                },
                function condition() {
                    return conditionFunction.call(self);
                },
                function exit(err) {
                    (err === null) ? resolve() : reject(err);
                });
    });
}

function promisify(scope, functionName, params) {
    return new Promise((resolve, reject) => {
        scope[functionName](params, (err, response) => {
            if (err) {
                return reject(err);
            }
            resolve(response);
        });
    });
}


/**
 * Calculates in percent, the change between 2 numbers.
 * e.g from 1000 to 500 = 50%
 * 
 * @param oldNumber The initial value
 * @param newNumber The value that changed
 */
function getPercentageChange(oldNumber, newNumber) {
    var decreaseValue = oldNumber - newNumber;
    const percentageChange = (decreaseValue / oldNumber) * 100;
    return percentageChange;
}


module.exports = {
    doUntil,
    promisify,
    getPercentageChange
};