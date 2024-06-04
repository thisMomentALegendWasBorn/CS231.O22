import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

train = pd.read_csv('./Project/blood_cell_training.csv')
train.head()

data = pd.DataFrame()
data['format'] = train['filename']

for i in range(data.shape[0]):
  data['format'][i] = 'JPEGImages/' + data['format'][i]

for i in range(data.shape[0]):
  data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['cell_type'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')
