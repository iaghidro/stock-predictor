const dataFetcher = require("../lib/dataFetcher/DataFetcher");

var execute = function() {
  console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

  // const pairs = ["ETH-USD"];
  const pairs = ["ETH-USD", "EOS-USD", "LTC-USD", "BTC-USD"];

  dataFetcher
    .fetchMulti(pairs)
    .then(processed => console.log(processed))
    .catch(err => {
      console.log("ERR");
      console.log(err);
    });
};

execute();
