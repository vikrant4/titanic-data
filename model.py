import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

def data_clean(d, p=False):
  scaler = MinMaxScaler()
  drop_columns = ['Embarked', 'Name', 'Ticket', 'Cabin']
  if p == False:
    drop_columns.append('PassengerId')  
  d.drop(columns=drop_columns, inplace=True)
  
  age_mask = d['Age'].isnull()
  d.loc[age_mask, 'Age'] = d['Age'].median()

  fare_mask = d['Fare'].isnull()
  d.loc[fare_mask, 'Fare'] = d['Fare'].mean()
  
  d['Sex'].replace('male', 1, inplace=True)
  d['Sex'].replace('female', 0, inplace=True)

  d[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = scaler.fit_transform(d[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
  return d

def predict(clf):

  test = pd.read_csv("test.csv")
  test = data_clean(test, True)

  target = 'Survived'

  predicted = clf.predict(test.drop('PassengerId', axis=1))
  predicted = pd.Series(predicted, name='Survived')

  prediction = pd.concat([test['PassengerId'], predicted], axis=1)

  prediction.to_csv("submission.csv", index=False)




def main():
  train = pd.read_csv("train.csv")
  train = data_clean(train)

  target = 'Survived'

  kf = KFold(n_splits=train.shape[0], random_state=4)

  clf = KNeighborsClassifier(5, weights='distance')
  fold_score = []

  for k, (train_index, test_index) in enumerate(kf.split(train)):
    print("Running fold", k)
    X_train, X_test = train.drop(target, axis=1).loc[train_index], train.drop(target, axis=1).loc[test_index]
    y_train, y_test = train.loc[train_index, target], train.loc[test_index, target]

    clf.fit(X_train, y_train)
    fold_score.append(clf.score(X_test, y_test))
  
  print("Average score", pd.Series(fold_score).mean())
  predict(clf)

if __name__ == '__main__':
    main()