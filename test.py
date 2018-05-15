from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

datasets = iris.data
labels = iris.target

print(datasets)

x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2)

num_cv = 10
lasso_model = linear_model.LassoCV(cv=num_cv).fit(x_train, y_train)

confident = lasso_model.score(x_test,y_test)

print(confident)


prediction = lasso_model.predict([6.7 ,3.0  ,5.2 ,2.3])
