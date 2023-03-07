import pandas as pd
# scikit-learn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split,RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#imblearn
from sklearn.pipeline import Pipeline
import joblib

def imb_pipe_fit(model, params,data_name, X, y, score='roc_auc', scaler=False):
    """
    utilize imblearn to create pipeline(smote->scale->model) and do gridsearch on model
    @params: model to fit,
             param_grid dictionary, 
             predictors(X), 
             response(y), 
             score_metric (sklearn), https://scikit-learn.org/stable/modules/model_evaluation.html
             scaler
    @return: dict of gridsearch best param model, cv_score, test_score
    """

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=421
                                                        )

    if scaler:
        pipeline = Pipeline(steps = [['scaler', StandardScaler()],
                                     ['model', model]]
                            )
    else:
        pipeline = Pipeline(steps = [['model', model]]
                            )
        
    folds = RepeatedStratifiedKFold(n_splits=10, 
                                    n_repeats=3,
                                    random_state=421
                                    )   
    print("fitting model...")
    gs = GridSearchCV(estimator=pipeline,
                    param_grid=params,
                    scoring=str(score),
                    cv=folds,
                    refit=True,
                    n_jobs=-1,
                    verbose=False
                    )     
    gs.fit(X_train, y_train)
    cv_score = gs.best_score_
    test_score = gs.score(X_test, y_test)
    joblib.dump(gs, f"models/{type(model).__name__}_{data_name}.pkl")

    return {'model':gs, 'cv_score':cv_score, 'test_score':test_score, 'best_param':gs.best_params_, "cv_results":gs.cv_results_}