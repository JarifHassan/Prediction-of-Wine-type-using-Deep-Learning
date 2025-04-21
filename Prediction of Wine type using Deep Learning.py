import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#2. Loading and Preprocessing Data

red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

red['type']=1
white['type']=0

wines = pd.concat([red,white], ignore_index=True)
wines.dropna(inplace=True)

#3. Plotting Distribution of Alcohol

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(wines[wines['type'] == 1].alcohol, bins=10, facecolor='red',alpha=.5, label = 'Red wine')
ax[1].hist(wines[wines['type'] == 0].alcohol, bins=10, facecolor='white',
edgecolor = 'black', lw=.5, alpha = .5, label= 'White wine')

for a in ax:
    a.set_ylim([0,1000])
    a.set_xlabel('Alcohol in % Vol')
    a.set_ylabel('Frequency')

ax[0].set_title('Alcohol Content in Red Wine')
ax[1].set_title('Alcohol Content in White Wine')

fig.suptitle('Distribution of Alcohol by Wine Type')
plt.tight_layout()
plt.show()

#4. Splitting Data into Training and Testing Sets

x= wines.iloc[:,:-1]
y = wines['type']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.34,
random_state=45)

#5. Creating Neural Network Model

model = Sequential()
model.add(Dense(12, activation = 'relu', input_dim= 12))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#6. Training the Model and Making Predictions

model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=1)
y_pred = model.predict(x_test)
print("Predictions: ", y_pred)

