const DataFetcher = require("../lib/dataFetcher/DataFetcher");

var execute = function() {
  console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

  const pairs = [
    "ETH-USD",
    "BCH-USD",
    "LTC-USD",
    "BTC-USD",
    "XRP-USD",
    "EOS-USD",
    "ETC-USD"
  ];
  const exchange = "coinbase";
  const lookback = 300;
  const dataFetcher = new DataFetcher();
  dataFetcher
    .fetchMulti(pairs, lookback, exchange)
    .then(() => console.log("FINISHED"))
    .catch(err => {
      console.log("ERR");
      console.log(err);
    });
};

execute();
