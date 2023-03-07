from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib
from PyALE import ale

def hyeni_pyale(df,df_ohe,xgb,ale_col):

    ohe = OneHotEncoder()
    coded_feature = pd.DataFrame(ohe.fit_transform(df[[ale_col]]).toarray(),columns=[x for i in ohe.categories_ for x in i])
    coded_feature

    features = df_ohe.columns
    X_feat_raw = df_ohe.drop(coded_feature.columns.to_list(), axis=1, inplace=False).copy()

    one_hot_encoder = OneHotEncoder().fit(df[[ale_col]])

    def onehot_encode(feat, ohe=one_hot_encoder):
        col_names = ohe.categories_[0]
        feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
        feat_coded.columns = col_names
        return feat_coded

    ale_eff = ale(
        X=pd.concat([X_feat_raw,df[ale_col]], axis=1),
        model=xgb.best_estimator_.named_steps.model,
        feature=[ale_col],
        encode_fun=onehot_encode,
        predictors=features,
)
    return ale_eff