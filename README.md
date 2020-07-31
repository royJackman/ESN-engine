# ESN-Engine

This is a general purpose engine for an [**Echo State Network**](http://www.scholarpedia.org/article/Echo_state_network), a type of recursive nerual network. The advantages of this network is the smaller number of weights to train, allowing for faster learning and active example understanding for larger deep networks. In addition, this engine was build to support **oscillatory computing**, a byproduct of a Hopfield architecture and intelligent network generating. 

![Echo State Network](esn.jpg)

## Usage

This repo was created to work similarly to [scikit-learn](http://scikit-learn.org/stable/), which is based on *training*, *fitting*, and *predicting*. Once the data has been imported and massaged into shape, the output weights need to be **trained** or quietly **fit** so that the ESN can **predict** more and more accurate outputs. That's about it, you train or fit to set the model to the data so you can predict the correct outputs for different inputs.

Creating an ESN: 

```myEsn = esn(50, 5, 5)```

Fitting:

```myEsn.fit(train_x, train_y)```

Predict over single input (overdampened):

```myEsn.predict(my_input, 0)```

Scoring:

```myEsn.score(test_x, test_y)```

ESN Engine v0.9