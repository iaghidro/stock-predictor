const createFile = require("create-file");
const app = require("../lib");
const axios = require("axios");
const moment = require("moment");
const { config } = app;

const uploadDataS3BucketName = process.argv[2]; //first param

const saveToFile = data => {
  const filePath = "/app/historicalData.json";
  return new Promise((resolve, reject) => {
    createFile(filePath, JSON.stringify(data), function(err) {
      if (err) {
        return reject(err);
      }
      resolve(data);
    });
  });
};

var execute = function() {
  console.log(`@@@@@@@@@ Start! @@@@@@@@@`);

  const apiKey = "d670934fe84f6b23e24f8e4436a89dc7";
  const baseURL = `https://api.nomics.com/v1/markets?key=${apiKey}`;

  axios
    .get(`${baseURL}&exchange=binance&base=BTC,ETH,LTC,XMR&quote=BTC,ETH,BNB`)
    .then(response => {
      console.log(response)
      saveToFile(response)
    })
    .catch(err => {
      console.log("ERRR");
      console.log(err);
    });
};

execute();
