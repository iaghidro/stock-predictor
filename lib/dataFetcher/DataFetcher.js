var fs = require("fs");
const moment = require("moment");
const _ = require("lodash");
const json2csv = require("json2csv");
const util = require("../util");
const { promisify } = require("util");
const maxDataWindow = 300;

class DataFetcher {
  async fetchMulti(pairs, lookbackWindow = 1, exchange) {
    console.log(`DataFetcher::fetchMulti`);
    const queries = [];
    pairs.map((pair, index) =>
      queries.push(
        util
          .delay(1000 * index)
          .then(() => this.fetchSinglePair(pair, lookbackWindow))
      )
    );
    const signleResponses = await Promise.all(queries);
    const multiResponse = [];
    signleResponses.map(res => multiResponse.push(...res));
    const csv = await this.converToCSV(multiResponse);
    return this.saveCSVToFile(csv, exchange);
  }

  async fetchSinglePair(pair, lookbackWindow = 1) {
    const requestsCount = Math.ceil(lookbackWindow / maxDataWindow);
    const requests = _.range(requestsCount);
    const queries = [];
    requests.map(count =>
      queries.push(
        util
          .delay(1500 * count)
          .then(() =>
            this.query(pair, this.createConfig(lookbackWindow, count))
          )
      )
    );
    const signleResponses = await Promise.all(queries);
    const multiResponse = [];
    signleResponses.map(res => multiResponse.push(...res));
    // console.log(`multiResponse: ` );
    // console.log(multiResponse);
    return this.processResponse(pair, multiResponse);
  }

  createConfig(lookbackWindow, count) {
    const now = moment().format();
    const start = moment(now)
      .subtract(lookbackWindow, "days")
      .format("YYYY-MM-DD HH:mm");
    const end = moment(now).format("YYYY-MM-DD HH:mm");
    console.log("start: " + start);
    console.log("end: " + end);
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

module.exports = DataFetcher;
