const dataFetcher = require("../lib/dataFetcher/DataFetcher");

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

  dataFetcher
    .fetchMulti(pairs, exchange)
    .then(() => console.log("FINISHED"))
    .catch(err => {
      console.log("ERR");
      console.log(err);
    });
};

execute();
