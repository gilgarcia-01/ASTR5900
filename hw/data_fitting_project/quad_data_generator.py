import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#define the model parameters for
# f(x) = a0 + a1*x + a2*x^2
a0 = 1.
a1 = -1.
a2 = 0.5

N = 50 #num of data points we want
x_vals = np.sort(10*np.random.rand(N)) #random list of x values
y_true = a0 + a1*x_vals + a2*x_vals**2 #true y values for given x vals

#we now displace the true y values by a random amount:
rng = np.random.default_rng()
displacement = rng.standard_normal(N)
#displacement = np.random.rand(N)
displacement_scalar = 2.
print(displacement)
y_data = y_true + (displacement_scalar * displacement)

#let's plot the fake data against the line
plt.scatter(x_vals,y_data,color='k',marker='+')
plt.plot(x_vals,y_true,color='k')
plt.show()


#lets export our fake data:
df_dict = {
'x': x_vals,
'y': y_data
}

df = pd.DataFrame.from_dict(df_dict)

print(df)
df.to_csv('data/data_question1.txt',sep='\t',index=False)








