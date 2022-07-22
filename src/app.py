import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
df = pd.read_csv(url)


#Transform categorical features

df = pd.get_dummies(df, columns=['sex','smoker','region'], drop_first=True)


x = df.drop("charges", axis=1)
y = df.charges

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.25, random_state = 15)



model = LinearRegression() 
model.fit(x_train, y_train)

poly_features = PolynomialFeatures(degree = 2) 

#Transform features

x_poly_train = poly_features.fit_transform(x_train) 
x_poly_test = poly_features.fit_transform(x_test)

model4 = model.fit(x_poly_train, y_train)

filename = 'models/model.sav'

pickle.dump(model4, open(filename,'wb'))


#load model

loaded_model = pickle.load(open(filename,'rb'))

