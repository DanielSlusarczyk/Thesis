import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import plotly.express as px
import numpy as np
import pandas as pd

class EvaluteModel():
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, X_validation: pd.DataFrame, county_names, random_state=None) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.X_validation = X_validation
        self.county_names = county_names
        self.random_day = random_state

    def test(self, model, random_day=None):
        self.results = pd.DataFrame()
        self.resultsPerCounty = pd.DataFrame(columns=['County', 'MAE', 'MSE'])

        self.y_pred = model.predict(self.X_test)

        self.__MAE()
        self.__MSE()
        self.__metricsPerCounty()
        display(self.results)
        display(self.resultsPerCounty)

        self.__Plot(random_day)
    
    def __metricsPerCounty(self):
        validationData = self.X_test[['county']].copy()
        validationData['predictions'] = self.y_pred.tolist()
        validationData['target'] = self.y_test

        data = validationData.groupby('county').mean().reset_index()

        for county in data.county.unique():
            countyData = data[data.county == county]

            mae = MAE(countyData['target'], countyData['predictions'])
            mse = MSE(countyData['target'], countyData['predictions'])

            self.resultsPerCounty.loc[len(self.resultsPerCounty.index)] = [self.county_names[county], mae, mse] 
            
    def __MAE(self):
        self.results['MAE'] = [MAE(self.y_pred, self.y_test)]

    def __MSE(self):
        self.results['MSE'] = [MSE(self.y_pred, self.y_test)]

    def __Plot(self, random_day = None):

        # Get random date for plot
        if random_day is not None:
            self.random_day = random_day
        else:
            self.random_day = self.X_validation['datetime_date'].sample(random_state=self.random_state).iloc[0]

        # Get week based on random date
        self.X_validation['target'] = self.y_test
        self.X_validation['predictions'] = self.y_pred.tolist()
        self.validationData = self.X_validation[
            (self.X_validation.datetime_date >= self.random_day - datetime.timedelta(days=3)) &
            (self.X_validation.datetime_date <= self.random_day + datetime.timedelta(days=3))
        ]

        if len(self.validationData) == 0:
            display(f'Warning: Test data in range {min(self.X_validation.datetime_date)} - {max(self.X_validation.datetime_date)}')

        validationData = self.validationData[['datetime', 'target', 'is_consumption', 'is_business', 'predictions']].copy()
        validationData = validationData.groupby(by=['datetime', 'is_consumption', 'is_business']).mean().reset_index()

        fig = px.line(validationData, x='datetime', y=['target', 'predictions'], facet_col='is_consumption', facet_row='is_business', title=f'Example')
        fig.show()