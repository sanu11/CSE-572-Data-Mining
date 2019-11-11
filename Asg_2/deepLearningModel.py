from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def build_network():
    # defining a neural network for classification

    nn_model = Sequential()
    input_size = 5
    nodes_layer1 = 8
    nodes_layer2 = 15
    nodes_layer3 = 1

    nn_model.add(Dense(nodes_layer1, input_dim=input_size, activation='relu', kernel_initializer='random_normal'))
    nn_model.add(Dense(nodes_layer2, activation='relu', kernel_initializer='random_normal'))
    nn_model.add(Dense(nodes_layer3, activation='sigmoid', kernel_initializer='random_normal'))

    # Compile the neural network
    # Using Adaptive moment estimation (RMSProp + Momentum )
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return nn_model


def train_neural_network(train_features,train_labels):
    print("Starting DeepLearning Training Model")
    print("Len of training set:",len(train_features))

    # train the model
    eps = 300
    batch = 12
    nn_model = build_network()
    # nn_model = KerasClassifier(build_fn=build_network, batch_size=batch, nb_epoch=eps)

    # Using K-Fold to get Samples
    kfold_accuracies = []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(train_features):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)
        X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
        Y_train, Y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        nn_model.fit(X_train,Y_train, epochs=eps , batch_size=batch)
        kfold_accuracies.append(nn_model.evaluate(X_test, Y_test)[1])

    # Evaluate model
    # accuracy = cross_val_score(estimator=nn_model, X=train_features, y=train_labels, cv=10, n_jobs=-1)
    # loss, accuracy = nn_model.evaluate(train_features, train_labels)
    # print("Accuracy: ", accuracy," Loss",loss)
    print("Accuracies: ",kfold_accuracies)
    print("Mean Accuracy: ", sum(kfold_accuracies)/len(kfold_accuracies))

    nn_model.save('nn_model.h5')


def test_neural_network(test_features, test_labels):
    nn_model = load_model('nn_model.h5')
    predictions = nn_model.predict_classes(test_features)
    predictions = (predictions > 0.5)
    print("\n\n***********Neural Network Output******************")
    print("\n**************************************************")
    print("Neural Network accuracy for test Data =  ", accuracy_score(test_labels, predictions))
    print("Neural Network precision for test Data =  ", precision_score(test_labels, predictions))
    print("Neural Network recall for test Data =  ", recall_score(test_labels, predictions))
    print("Neural Network F1 for test Data =  ", f1_score(test_labels, predictions))

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix [[TT, TF],[FT, FF] ]", cm)
    return predictions

def nn_test_one_sample(test_data):
    nn_model = load_model('nn_model.h5')
    prediction = nn_model.predict(test_data)
    prediction = (prediction > 0.5)
    return prediction