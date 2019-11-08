from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def train_neural_network(train_features,train_labels):
    print("Starting DeepLearning Training Model")
    # defining a neural network for classification
    nn_model = Sequential()
    input_size = 5
    nodes_layer1 = 30
    nodes_layer2 = 15
    nodes_layer3 = 1

    nn_model.add(Dense(nodes_layer1, input_dim=nn_model, activation='relu',kernel_initializer='random_normal'))
    nn_model.add(Dense(nodes_layer2, activation='relu',kernel_initializer='random_normal'))
    nn_model.add(Dense(nodes_layer3, activation='sigmoid',kernel_initializer='random_normal'))

    # Compile the neural network
    # Using Adaptive moment estimation (RMSProp + Momentum )
    nn_model.compile(optimizer='adam', loss='binary_crossentropy',metrics =['accuracy'])

    # train the model
    eps = 144
    batch = 12
    nn_model.fit(train_features,train_labels, epochs=eps , batch_size=batch)

    # Evaluate model
    loss, accuracy = nn_model.evaluate(train_features, train_labels)
    print("Accuracy: "+ accuracy," Loss",loss)

    nn_model.save('nn_model.h5')


def test_neural_network(test_features, test_labels):
    nn_model = load_model('nn_model.h5')
    predictions = nn_model.predict_classes(test_features)
    predictions = (predictions > 0.5)
    print("Accuracy for test Data: ",accuracy_score(test_labels, predictions))

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix [[TT, TF],[FT, FF] ]", cm)
    return predictions