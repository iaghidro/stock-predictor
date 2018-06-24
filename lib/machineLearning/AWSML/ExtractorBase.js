const _ = require('lodash');

class ExtractorBase {

    getCurrentFeatureSet() {
        return this.list[this.index];
    }

    /**
     * Returns previous feature set in list. If current item is first, it returns current item.
     * @returns {Object} FeatureSet
     */
    getPreviousFeatureSet() {
        const previousIndex = this.index - 1;
        if (previousIndex < 0) {
            return this.getCurrentFeatureSet();
        }
        return this.list[previousIndex];
    }

    /**
     * Returns next feature set in list. If current item is last, it returns current item.
     * @returns {Object} FeatureSet
     */
    getNextFeatureSet() {
        const nextIndex = this.index + 1;
        if (nextIndex > this.list.length) {
            return this.getCurrentFeatureSet();
        }
        return this.list[nextIndex];
    }
    
    /**
     *  Returns a subset of this.list determined by the lookbackPeriod.
     * @param {Integer} lookbackPeriod the number of indexes to look back starting from the current index
     * @returns {Array} The list of values for the determined range
     */
    getRange(lookbackPeriod) {
        const range = this.index - lookbackPeriod;
        const periodStart = range >= 0 ? range : 0;
        const periodEnd = this.index + 1;
        return this.list.slice(periodStart, periodEnd);
    }
    
    /**
     * Returns a subset of this.list and only for the given attribute name determined by the lookbackPeriod.
     * 
     * @param {Integer} lookbackPeriod the number of indexes to look back starting from the current index
     * @param {type} attributeName The attribute to choose from each feature set
     * @returns {Array} The list of values specified by the attribute name
     */
    getRangeValues(lookbackPeriod, attributeName) {
        const historicalData = this.getRange(lookbackPeriod);
        return  _.map(historicalData, attributeName);
    }
    
}

module.exports = ExtractorBase;