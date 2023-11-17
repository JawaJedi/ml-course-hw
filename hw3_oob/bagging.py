from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import numpy as np
from random import choices

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            self.indices_list.append(choices(range(data_length), k = data_length)) #сэмплирование с возвратом

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(
            data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            #data_bag, target_bag =  0# Your Code Here
            data_bag = data[self.indices_list[bag],:]
            target_bag = target[self.indices_list[bag]]
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        res = []
        for model in self.models_list:
            res.append(model.predict(data))
        return np.average(res, axis=0)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        i=0
        for prediction_list in list_of_predictions_lists:
            for bag in range(self.num_bags):
                if i in self.indices_list[bag]:
                    prediction_list.append(self.models_list[bag].predict([self.data[i]]))
            i += 1
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        oob_predictions = []
        for prediction_list in self.list_of_predictions_lists:
            if len(prediction_list) < self.num_bags:
                oob_predictions.append(np.average(prediction_list))
            else:
                oob_predictions.append(None)
        self.oob_predictions = oob_predictions # Your Code Here

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        oob_scores = []
        for i in range(len(self.target)):
            if self.oob_predictions[i] != None:
                oob_scores.append(np.square(self.oob_predictions[i] - self.target[i]))
        return np.average(oob_scores)# Your Code Here