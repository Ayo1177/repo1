#import piplite
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler,PolynomialFeatures




file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'

df = pd.read_csv(file_name)
#q1
#print(df.dtypes)
#q2
df.drop(['Unnamed: 0','id'],axis=1,inplace=True)
#print(df.head(2))
#print(df.describe())
df['bedrooms'].replace(np.nan, df['bedrooms'].mean(), inplace=True)
df['bathrooms'].replace(np.nan, df['bathrooms'].mean(), inplace=True)
#sns.boxplot(x='waterfront',y='price',data=df)
#plt.xlabel('Waterfront')
#plt.ylabel('Price')
#plt.title('Box Plot of Price by Waterfront')
#plt.show()
#print(df[['floors']].value_counts().to_frame())
#qu5
#sns.regplot(x='sqft_above',y='price',data=df)
#plt.xlabel('sqft_above')
#plt.ylabel('price')
#plt.title('correlation between price and square feet')
#plt.show()
# Check correlation
from sklearn.linear_model import LinearRegression

#X = df[['sqft_living']]
#Y = df[['price']]
#lm.fit(X, Y)
#print("the R^2 value is", lm.score(X, Y))


#features = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']].values
#target = df['price'].values
#lm = LinearRegression()
#lm.fit(features, target)
#print('the R^2 value is', lm.score(features, target))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
#pipe = Pipeline(Input)
#features = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']].values
#target = df['price'].values
#pipe.fit(features, target)
#print('the R^2 value is', pipe.score(features, target))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
print("done")
X = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']].values
#X = df['features']
Y = df['price'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
RM = Ridge(alpha=0.1)
RM.fit(X, Y)
print('the R^2 value is', RM.score(X, Y))

#from sklearn.model_selection import train_test_split²
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import Ridge
#pr = PolynomialFeatures(degree=2)
#X = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']].values
#Y = df['price'].values
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
#x_train_pr = pr.fit_transform(x_train)
#x_test_pr = pr.fit_transform(x_test)
#poly = Ridge(alpha=0.1)
#poly.fit(x_train_pr, y_train)
#print(poly.score(x_test_pr, y_test))
