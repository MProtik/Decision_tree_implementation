import pandas as pd
df =pd.read_csv('shop data.csv')
df
x=df.iloc[:,:-1]
x
y=df.iloc[:,-1]
y

from sklearn.preprocessing import LabelEncoder
Le_x=LabelEncoder()
x=x.apply(LabelEncoder().fit_transform)
x

from sklearn.tree import DecisionTreeClassifier
import numpy as np
dtf=DecisionTreeClassifier()
dtf.fit(x.iloc[:,0:4],y)
xinput=np.array([1,0,1,1])
y_predict=dtf.predict([xinput])
y_predict

