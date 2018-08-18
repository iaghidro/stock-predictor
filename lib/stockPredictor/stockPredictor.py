import numpy as np
import pandas as pd
#from fastai.structured import *
#from fastai.column_data import *
#np.set_printoptions(threshold=50, edgeitems=20)
from ta import *

class StockPredictor:
    
    def __init__(self, df, index):
        self.df = df
        self.index = index        
                
    # ///////////////////////////////
    # /////// DATA CLEANING /////////
    # ///////////////////////////////
    
    def sample_train(self, sampleSize):
        self.train = self.df.tail(sampleSize)
        print('StockPredictor::sample_train:: Train size: ' + str(len(self.train)) + ' Original size: ' + str(len(self.df)))
        
    def set_date_as_index(self):
        self.train[self.index] = pd.to_datetime(self.train[self.index], unit='s')
        
    def split_train_validation(self, testRecordsCount, trainRecordsCount):
        self.test = self.df.tail(testRecordsCount)
        self.train = self.df.head(trainRecordsCount)
#        self.test.reset_index(inplace=True)
#        self.train.reset_index(inplace=True)
        print('StockPredictor::split_train_validation:: Train size: ' + str(len(self.train)) + ' Test size: ' + str(len(self.test)))    

    def add_ta(self):
        self.train = add_all_ta_features(self.train, "Open", "High", "Low", "Close", "Volume", fillna=True)
        
    def clean_train(self):
    #     df = df.dropna()
    #     df = df.replace(np.nan,df.mean())
        self.train = self.train.replace([np.inf, -np.inf], np.nan)
        self.train = self.train.fillna(method='bfill')
        

    # ///////////////////////////////
    # ///////// EVALUATION //////////
    # ///////////////////////////////
    
    def calculate_accuracy(self, df):
        successful_predictions = df.loc[df.action == df.predicted]
        total_accuracy = len(successful_predictions)/len(df)
        total_buy_actions = df.loc[df.action == 1]
        total_sell_actions = df.loc[df.action == 0]
        successful_buy_predictions = successful_predictions.loc[successful_predictions.action == 1]
        successful_sell_predictions = successful_predictions.loc[successful_predictions.action == 0]
        buy_accuracy = len(successful_buy_predictions)/len(total_buy_actions)
        sell_accuracy = len(successful_sell_predictions)/len(total_sell_actions)
        f1Score = (buy_accuracy + sell_accuracy)/2;
        result = {
            'F1Score': round(f1Score,3),
            'totalAccuracy': round(total_accuracy,3),
            'buyAccuracy': round(buy_accuracy,3),
            'sellAccuracy': round(sell_accuracy,3),
            'totalBuyActions': len(total_buy_actions),
            'successfulBuyPredictions': len(successful_buy_predictions)
        }
        return result
            
    def calculate_net_profit(self, inputDf, startAmount):
        df = inputDf
        df['buyAmount'] = 0
        df['sellAmount'] = 0
        totalBuys = 0
        totalSells = 0
        for index, row in df.iterrows():
            prevBuyAmount = df.buyAmount.get(index -1, np.nan)
            prevSellAmount = df.sellAmount.get(index -1, np.nan)
    #         prevPredicted = df.predicted.get(index -1, np.nan)
            predicted = row.predicted
            if index == df.index[0]:
                df.loc[index,'buyAmount'] = startAmount
            elif predicted == 1 and prevBuyAmount > 0:
                # BUY
                df.loc[index,'sellAmount'] = prevBuyAmount/row.Close
                totalBuys +=1
            elif predicted == 1 and prevBuyAmount == 0:
                df.loc[index,'sellAmount'] = prevSellAmount
            elif predicted == 0 and prevSellAmount > 0:
                # SELL             
                df.loc[index,'buyAmount'] = prevSellAmount*row.Close
                totalSells +=1
            elif predicted == 0 and prevSellAmount == 0:
                df.loc[index,'buyAmount'] = prevBuyAmount
            else:
                # HOLD (not holding currently)
                df.loc[index,'buyAmount'] = prevBuyAmount
                df.loc[index,'sellAmount'] = prevSellAmount
                
        startClose = df.Close.iloc[0]
        endClose = df.Close.iloc[-1]
        endBuyAmount = df.buyAmount.iloc[-1]
        endSellAmount = df.sellAmount.iloc[-1]
        endAmount = endBuyAmount if (endBuyAmount > 0) else (endSellAmount * endClose)
        buyAndHoldPercentIncrease = ((endClose - startClose)/startClose) * 100
        percentIncrease = ((endAmount - startAmount)/startAmount) * 100
        percentDifference = percentIncrease - buyAndHoldPercentIncrease

        result = {
            'startClose': startClose,
            'endClose': endClose,
            'startAmount': startAmount,
            'endAmount': endAmount,
            'buyAndHoldPercentIncrease':round(buyAndHoldPercentIncrease,3),
            'percentIncrease':round(percentIncrease,3),
            'percentDifference':round(percentDifference,3),
            'totalTrades':totalBuys + totalSells
        }
        return result

    # ///////////////////////////////
    # /////////// UTIL //////////////
    # ///////////////////////////////
    
    def save_to_feather(self):
        self.train.reset_index(inplace=True)
        self.train.to_feather(f'{PATH}train')
    
    def read_from_feather(self):
        self.train = pd.read_feather(f'{PATH}train')
        train.drop(self.index,1,inplace=True)
    
    """ usage conflateTimeFrame(df, '5T') """
    def conflate_time_frame(self, df, timeFrame):
        ohlc_dict = {                                                                                                             
            'Open':'first',                                                                                                    
            'High':'max',                                                                                                       
            'High':'max',                                                                                                       
            'Low':'min',                                                                                                        
            'Close': 'last',                                                                                                    
            'Volume': 'sum'
        }
        return df.resample(timeFrame).agg(ohlc_dict)