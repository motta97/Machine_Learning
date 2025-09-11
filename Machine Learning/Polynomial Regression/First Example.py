import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x= 6*np.random.rand(100,1)-3
y = x**2+ x+ 10 +np.random.rand(100,1)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
x_poly=poly.fit_transform(x)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
x_test_poly = poly.transform(x_test)
y_test=lin_reg.predict(x_test_poly)


plt.scatter(x,y,color="blue")
plt.plot(x_test,y_test,"-r")
plt.show()
