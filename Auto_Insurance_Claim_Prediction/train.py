import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv("https://raw.githubusercontent.com/vandithavb/machine-learning-zoomcamp-2025/refs/heads/main/Auto_Insurance_Claim_Prediction/car_insurance_claim.csv")


# Cleaning columns (Currency to Numeric)
currency_cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
for col in currency_cols:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)   


# Dropping below columns as it is not relevant
df=df.drop(['ID', 'BIRTH'], axis = 1)


num = ['KIDSDRIV',
 'AGE',
 'HOMEKIDS',
 'YOJ',
 'INCOME', 
 'HOME_VAL',
 'TRAVTIME',
 'BLUEBOOK',
 'TIF',
 'OLDCLAIM',
 'CLM_FREQ',
 'MVR_PTS',
 'CLM_AMT',
 'CAR_AGE',
 ]
cat = ['PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'REVOKED', 'URBANICITY']


df[num] = df[num].fillna(df[num].median())
df[cat] = df[cat].fillna('NA')



# Setting up the Validation Framework
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state =1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state =1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


y_train = df_train.CLAIM_FLAG.values
y_val = df_val.CLAIM_FLAG.values
y_test = df_test.CLAIM_FLAG.values
y_full_train = df_full_train.CLAIM_FLAG.values

del df_train['CLAIM_FLAG']
del df_val['CLAIM_FLAG']
del df_test['CLAIM_FLAG']


#Feature Selection

features = [
    'KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL',
    'PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION', 'OCCUPATION',
    'TRAVTIME', 'CAR_USE', 'BLUEBOOK', 'TIF', 'CAR_TYPE', 'RED_CAR',
    'OLDCLAIM', 'REVOKED', 'MVR_PTS', 'CAR_AGE', 'URBANICITY'
    # note: no CLM_AMT, no CLM_FREQ
]



# Preparing data
dv = DictVectorizer(sparse=False)
train_dicts = df_full_train[features].to_dict(orient='records')
X_full_train = dv.fit_transform(train_dicts)
test_dicts = df_test[features].to_dict(orient='records') 
X_test = dv.transform(test_dicts)


# Train the model
rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=1,
            n_jobs=-1
        )
rf.fit(X_full_train, y_full_train)
y_test_pred = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_test_pred)



# Save the model
output_file = 'model.bin'
with open(output_file, 'wb') as f_out:  
    pickle.dump((dv, rf), f_out)

