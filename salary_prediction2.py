# This file is a copy of the code run on google colab
# !pip install category-encoders

import time
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
import pandas_profiling
from sklearn.metrics import mean_squared_error, mean_absolute_error
import category_encoders as ce
import matplotlib.pyplot as plt

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import regularizers
from keras.callbacks import ModelCheckpoint

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

TRAIN_FILE = '/content/drive/My Drive/Colab Notebooks/IncomePredGroup/train.csv'
TEST_FILE = '/content/drive/My Drive/Colab Notebooks/IncomePredGroup/test.csv'
OUTPUT_FOLDER = '/content/drive/My Drive/Colab Notebooks/IncomePredGroup/output/'

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


def clean_price(price_str):
    price = float(price_str.replace(' EUR',''))
    return price


def make_data_consistent(dataset):
    d = dataset.copy()
    d['Housing Situation'] = d['Housing Situation'].replace(0,np.nan).replace('0',np.nan).replace('nA',np.nan)
    d['Gender'] = d['Gender'].replace('0', np.nan).replace('unknown',np.nan).replace('f','female')
    d['University Degree'] = d['University Degree'].replace('0',np.nan)
    d['Hair Color'] = d['Hair Color'].replace('0',np.nan).replace('Unknown',np.nan)
    d['Work Experience in Current Job [years]'] = d['Work Experience in Current Job [years]'].replace('#NUM!',np.nan).astype(float)
    d['Country'] = d['Country'].replace('0',np.nan)
    d['Yearly Income in addition to Salary (e.g. Rental Income)'] = [clean_price(y) for y in d['Yearly Income in addition to Salary (e.g. Rental Income)']]
    # analyse_df(d)
    return d


def populate_missing_data(dataset):
    d = dataset.copy()
    d['Year of Record'].fillna(d['Year of Record'].mean(),inplace=True)
    d['Housing Situation'].fillna('Unknown',inplace=True)
    d['Age'].fillna(d['Age'].mean(),inplace=True)
    d['Work Experience in Current Job [years]'].fillna(d['Work Experience in Current Job [years]'].mean(),inplace=True)
    d['Satisfation with employer'].fillna(method='bfill',inplace=True)
    d['Profession'].fillna(method='bfill',inplace=True)
    d['Gender'].fillna('Unknown',inplace=True)
    d['Hair Color'].fillna('Unknown',inplace=True)
    d['University Degree'].fillna('Unknown',inplace=True)
    return d


def analyse_df(df):
    res = {}
    for col in df.columns:
        uniqueVals = df[col].unique()
        uniqueValsCount = len(uniqueVals)
        nanCount = df[col].isna().sum()*100/991709
        res[col] = {
            'unique_count':uniqueValsCount,
            # 'uniqueVals':uniqueVals
            'nan_count':nanCount
        }
        print(col + ' :: ' + str(res[col]))


def encode_categories(dataset,testDataset):
    train = dataset.copy()
    test = testDataset.copy()
    targetCatColumns = ['Profession', 'Country']
    ce_targetEncoder = ce.TargetEncoder()
    train[targetCatColumns] = ce_targetEncoder.fit_transform(train[targetCatColumns], train['Total Yearly Income [EUR]'])
    test[targetCatColumns] = ce_targetEncoder.transform(test[targetCatColumns])

    catColumns = ['Gender','University Degree','Hair Color','Housing Situation','Satisfation with employer']
    train['train'] = 1
    test['train'] = 0
    combined = pd.concat([train,test],sort=False)
    df = pd.get_dummies(combined,columns=catColumns,prefix=catColumns)
    train = df[combined['train'] == 1]
    test = df[combined['train'] == 0]
    train.drop(['train'],axis=1,inplace=True)
    test.drop(['train','Total Yearly Income [EUR]'],axis=1,inplace=True)
    return train,test


def scale_data(X):
    stScaler_ds = StandardScaler()
    X = stScaler_ds.fit_transform(X)
    return X


def split_data(X, y):
    # split data into training and test parts
    testSize = 0.05
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=testSize)
    return (Xtrain, Xtest, ytrain, ytest)


def plot_corr_matrix(dataset):
    df = dataset.copy(deep=True)
    # df = df[df.columns.drop(list(df.filter(regex='Profession')))]
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    fig, ax = plt.subplots(figsize=(df.columns.size, df.columns.size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('plot.png')
    plt.show()


def predict_data(model, dataset, featureCol, outputFile):
    print("Predicting values")
    cols = featureCol.copy()
    cols = cols[cols!='Total Yearly Income [EUR]']
    instanceIds = dataset['Instance'].values
    data = dataset.loc[:, cols]

    X = data.values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    predictihttps://github.com/ = model.predict(X).flatten()
    # predichttps://github.com/on = np.expm1(prediction)
    result = pd.DataFrame({'Instance':instanceIds, 'Total Yearly Income [EUR]':prediction})
    result.to_csv(outputFile,index=False)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()


def main():
    dataset = pd.read_csv(TRAIN_FILE)
    testDataset = pd.read_csv(TEST_FILE)

    dataset = make_data_consistent(dataset)
    testDataset = make_data_consistent(testDataset)

    dataset = populate_missing_data(dataset)
    testDataset = populate_missing_data(testDataset)

    # dataset['Total Yearly Income [EUR]'] = np.log1p(dataset['Total Yearly Income [EUR]'])

    # converting profession to lower case profession
    dataset['Profession'] = [x.lower() for x in dataset['Profession']]
    testDataset['Profession'] = [x.lower() for x in testDataset['Profession']]

    (dataset,testDataset) = encode_categories(dataset,testDataset)
    # analyse_df(dataset)
    # analyse_df(testDataset)
    print("Data pre-processing done")
    # plot_corr_matrix(dataset)

    ## ========================== dataset and testDataset are pd dataframe with clean data =========================== ##

    featureColumns = np.array(dataset.columns.drop('Instance'))
    data = dataset.loc[:, featureColumns]
    X = data[data.columns.drop('Total Yearly Income [EUR]')].values
    y = data['Total Yearly Income [EUR]'].values
    (Xtrain, Xtest, ytrain, ytest) = split_data(X, y)
    print('Xtrain size:: ' + str(len(Xtrain)))
    print('Xtest size:: ' + str(len(Xtest)))

    # Scale Data
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("No of Columns:: " + str(len(X[0])))

    model = keras.Sequential([
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_shape=[len(X[0])]),
        layers.Dropout(0.4),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1)
    ])

    model.compile(loss='mae',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    print(model.summary())
    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    gpus = tf.config.experimental.list_logical_devices('GPU')
    if gpus:
      # Replicate your computation on multiple GPUs
      c = []
      for gpu in gpus:
        with tf.device(gpu.name):
          print("fitting model..")
          history = model.fit(Xtrain, ytrain, epochs=50, validation_split=0.1, verbose=2,callbacks=[checkpoint,early_stop,PrintDot()])
          print("fit model")

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    # loss, mae, mse = model.evaluate(Xtest, ytest, verbose=2)
    ypred = model.predict(Xtest).flatten()
    ypred = np.expm1(ypred)
    ytestInv = np.expm1(ytest)
    mae = mean_absolute_error(ytestInv, ypred)
    print("Testing set Mean Abs Error: {:5.2f}".format(mae))
    date_time_stamp = time.strftime('%Y%m%d%H%M%S')  #in the format YYYYMMDDHHMMSS
    predict_data(model,testDataset,featureColumns, OUTPUT_FOLDER + 'tfOut-' + date_time_stamp + "-" + str(int(mae)) + ".csv")
    plot_history(history)


if __name__ == '__main__':
  main()
