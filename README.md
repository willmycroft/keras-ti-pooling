# A Keras network wrapper for implementing transformation-invariant pooling.

A Keras interface to the techniques detailed in *TI-POOLING: transformation-invariant pooling for feature learning in Convolutional Neural Networks* by Dmitry Laptev, Nikolay Savinov, Joachim M. Buhmann and Marc Pollefeys.

## Quick start
```python
from keras.models import Sequential, Model
from keras.layers import Dense
from ti_pooling import transformation_invariant
from ti_pooling.transformations import SymmetricGroup

transformation = SymmetricGroup()
network = Sequential()
network.add(Dense(3, activation='relu', input_shape=(10,))
network.add(Dense(1, activation='linear'))
inputs, output = transformation_invariant(network, transformation)
ti_network = Model(inputs=inputs, outputs=output)
ti_network.compile(loss='mse', optimizer='adam')
ti_network.fit(transformation.transform(X), y)
```

See the examples for more details.