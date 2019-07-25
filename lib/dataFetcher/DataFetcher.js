const moment = require("moment");
const util = require("../util");

class DataFetcher {
  async fetchMulti(pairs) {
    const queries = [];
    pairs.map((pair, index) => queries.push(this.fetch(pair, 1000 * index)));
    const signleResponses = await Promise.all(queries);
    const multiResponse = [];
    signleResponses.map(res => multiResponse.push(...res));
    return multiResponse;
  }

  async fetch(pair, delay = 0) {
    await util.delay(delay);
    const historicalRows = await this.query(pair, this.createConfig());
    const processed = await this.processResponse(pair, historicalRows);
    return processed;
  }

  createConfig() {
    const lookbackWindow = 3;
    const now = moment().format();
    const start = moment(now)
      .subtract(lookbackWindow, "days")
      .format("YYYY-MM-DD HH:mm");
    const end = moment(now).format("YYYY-MM-DD HH:mm");
    // console.log("start: " + start);
    // console.log("end: " + end);
    return {
      granularity: 86400,
      start,
      end
    };
  }

  query(pair, config) {
    const CoinbasePro = require("coinbase-pro");
    const publicClient = new CoinbasePro.PublicClient();
    return publicClient.getProductHistoricRates(pair, config);
  }

  processResponse(pair, historicalRows) {
    const historicalObject = [];
    historicalRows.forEach(row => {
      historicalObject.push({
        date: moment.unix(row[0]).format("YYYY-MM-DD HH:mm"),
        low: row[1],
        high: row[2],
        open: row[3],
        close: row[4],
        volume: row[4],
        symbol: pair
      });
    });
    return historicalObject;
  }
}

module.exports = new DataFetcher();
