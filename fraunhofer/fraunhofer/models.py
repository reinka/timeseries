# -*- coding: utf-8 -*-
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import sklearn

class KNN(object):
    def __init__(self, X, y, ):
        pass

    def accuracy_for_k(k, X_train, X_test, y_train, y_test):
        #     split_data=sklearn.cross_validation.train_test_split(x,y,test_size=0.33,random_state=99)
        #     X_train,X_test,Y_train,Y_test=split_data
        knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k,
                                                    weights='uniform')
        knn.fit(X_train, y_train)
        # Y_hat=knn.predict(X_test)
        value = knn.score(X_test, y_test)
        return value