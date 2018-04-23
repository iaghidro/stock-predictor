const _ = require('lodash');
const util = require('./../util');
const AWSSDKBase = require('./../aws/AWSSDKBase');

class S3 extends AWSSDKBase {

    constructor(params = {}) {
        super(params);
        this.s3 = this.createServiceInstance('S3');
    }

    uploadFile(params = {}) {
        const {bucketName, file, fileName} = params;
        var params = {
            Bucket: bucketName,
            Key: fileName,
            Body: file
        };
        var options = {
            partSize: 10 * 1024 * 1024,
            queueSize: 1
        };
        return new Promise((resolve, reject) => {
            this.s3.upload(params, options, (err, data) => {
                if (err) {
                    this.log.info(`S3::uploadFile failed`);
                    reject(err);
                } else {
                    this.log.info(`S3::uploadFile success`);
                    resolve();
                }
            });
        });
    }

    /////////////////////////////////
    ///////////  PRIVATE  ///////////
    /////////////////////////////////


}

module.exports = S3;
