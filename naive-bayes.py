import numpy as np 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelBinarizer

df = pd.read_csv('tabel_beli_laptop.csv')

umur = df['umur'].values.reshape(-1,1)
gaji = df['gaji'].values.reshape(-1,1)
status = df['status'].values.reshape(-1,1)
hutang = df['hutang'].values.reshape(-1,1)

umur_enc = OrdinalEncoder()
umur_ = umur_enc.fit_transform(umur)

gaji_enc = OrdinalEncoder()
gaji_ = gaji_enc.fit_transform(gaji)

status_enc = OrdinalEncoder()
status_ = status_enc.fit_transform(status)

hutang_enc = OrdinalEncoder()
hutang_ = hutang_enc.fit_transform(hutang)

# Stacking all the features
X = np.column_stack((umur_, gaji_, status_, hutang_))
# Changing the type to int
X = X.astype(int)

# Doing one hot encoding on the target data
y_ = df['beli_laptop']
lb = LabelBinarizer()
y = lb.fit_transform(y_)
if y.shape[1] == 1:
    y = np.concatenate((1 - y, y), axis=1)
classes = lb.classes_

n_features = X.shape[1]
n_classes = y.shape[1]
count_matrix = []

for i in range(n_features):
    count_feature = []
    X_feature = X[:,i]

    for j in range(n_classes):
        mask = y[:,j].astype(bool)
        counts = np.bincount(X_feature[mask])
        count_feature.append(counts)
        
    count_matrix.append(np.array(count_feature))
    
class_count = y.sum(axis=0)

num = class_count
den = class_count.sum()
class_probs = num/den

feat_probs = []
for i in range(n_features):
    num = count_matrix[i]
    den = num.sum(axis = 1).reshape(-1,1)
    probability = num/den
    feat_probs.append(probability)

query_point = [1, 1, 0, 0]
probs = np.ones((1,n_classes))

for i in range(n_features):
    category = query_point[i]
    probs*=feat_probs[i][:,category]

probs*=class_probs
predict = classes[np.argmax(probs)]

print('X = ', query_point)
print('P( y =', classes[0], ') = ', probs[0][0])
print('P( y =', classes[1], ') = ', probs[0][1])
print('Apakah beli laptop ? = ', predict)