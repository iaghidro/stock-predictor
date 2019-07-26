var fs = require('fs');
const moment = require("moment");
const json2csv = require("json2csv");
const util = require("../util");
const { promisify } = require("util");

class DataFetcher {
  async fetchMulti(pairs, exchange) {
    console.log(`DataFetcher::fetchMulti`);
    const queries = [];
    pairs.map((pair, index) => queries.push(this.fetch(pair, 1000 * index)));
    const signleResponses = await Promise.all(queries);
    const multiResponse = [];
    signleResponses.map(res => multiResponse.push(...res));
    const csv = await this.converToCSV(multiResponse);
    return this.saveCSVToFile(csv, exchange);
  }

  async fetch(pair, delay = 0) {
    await util.delay(delay);
    const historicalRows = await this.query(pair, this.createConfig());
    const processed = await this.processResponse(pair, historicalRows);
    return processed;
  }

  createConfig() {
    const lookbackWindow = 300;
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
    console.log(`DataFetcher::query ${pair}`);
    const CoinbasePro = require("coinbase-pro");
    const publicClient = new CoinbasePro.PublicClient();
    return publicClient.getProductHistoricRates(pair, config);
  }

  processResponse(pair, historicalRows) {
    const historicalObject = [];
    historicalRows.forEach(row => {
      historicalObject.push({
        Date: moment.unix(row[0]).format("YYYY-MM-DD HH:mm"),
        Low: row[1],
        High: row[2],
        Open: row[3],
        Close: row[4],
        Volume: row[4],
        Symbol: pair
      });
    });
    return historicalObject;
  }

  async converToCSV(data) {
    console.log(`DataFetcher::converToCSV`);
    let csv;
    const params = {
      data
    };
    try {
      csv = json2csv(params);
    } catch (err) {
      return Promise.reject(err);
    }
    return csv;
  }

  saveCSVToFile(data, fileName) {
    console.log(`ModelTrainer::saveCSVToFile`);
    const filePath = `/app/data/stock/${fileName}.csv`;
    return promisify(fs.writeFile)(filePath, data);
  }
}

module.exports = new DataFetcher();
