import pandas
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,normalize


def loaddata(pickle_file=True):
    if pickle_file:
        data_fid = open('data_svm.pkl', 'rb')
        [dataX, labels] = pickle.load(data_fid)
        data_fid.close()
    else:
        dataX = []
        dataframe = pandas.read_csv("Dataset_Lab.csv")
        dataset = dataframe.values
        X = dataset[:, 1:13]  # 1: 12 input data
        Y = dataset[:, 13:]   # 13, 14 output

        x0 = []
        for x in X[:, 0]:
            item = [float(x)]
            x0.append(np.asarray(item))
        x0 = np.array(x0, dtype=np.float32)
        dataX = x0

        x1 = []
        for x in X[:, 1]:
            item = [float(x)]
            x1.append(np.asarray(item))
        x1 = np.array(x1, dtype=np.float32)
        dataX = np.concatenate([dataX, x1], axis=1)

        for i in range(2,X.shape[1]):
            le = LabelEncoder()
            le.fit(X[:, i])
            xi = le.transform(X[:, i])
            xii = []
            for x in xi:
                item = [float(x)]
                xii.append(np.asarray(item))
            xii = np.array(xii, dtype=np.float32)
            dataX = np.concatenate([dataX, xii], axis=1)
        labels = np.array([np.argmax(y) for y in Y])

        with open('data_svm.pkl', 'wb') as file_id:
            pickle.dump([dataX, labels], file_id)
        file_id.close()

    return dataX, labels

X, y = loaddata(pickle_file=False)
X = normalize(X, norm='max', axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
# save the model to disk
filename = 'SVM_model_linear.sav'
pickle.dump(svclassifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)
print("Predicted")
print(y_pred)
print("Referance")
print(y_test)
result = loaded_model.score(X_test,y_test)
print("Score")
print(result)
