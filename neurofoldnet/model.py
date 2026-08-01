from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NeuroFoldNet(BaseEstimator, ClassifierMixin):


    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.layer1_models = []
        self.layer2_models = []
        self.meta_model = None
        self.layer1_final = []
        self.layer2_final = []

    def _get_base_models(self):
        return [
            (
                "xgb_deep",
                xgb.XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                ),
            ),
            (
                "xgb_wide",
                xgb.XGBClassifier(
                    n_estimators=90,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.05,
                    reg_lambda=0.8,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                ),
            ),
            (
                "gb_tuned",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    random_state=self.random_state,
                ),
            ),
            (
                "svm_rbf",
                SVC(C=8.0, kernel="rbf", gamma="scale", probability=True, random_state=self.random_state),
            ),
            (
                "svm_poly",
                SVC(C=5.0, kernel="poly", degree=3, probability=True, random_state=self.random_state),
            ),
        ]

    def _get_layer2_models(self):
        return [
            (
                "xgb_meta",
                xgb.XGBClassifier(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                ),
            ),
            (
                "gb_meta",
                GradientBoostingClassifier(
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.08,
                    random_state=self.random_state,
                ),
            ),
        ]

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        n_classes = len(self.classes_)
        inner_folds = min(self.n_folds, np.bincount(y).min())
        if inner_folds < 2:
            raise ValueError("NeuroFoldNet requires at least two examples per class in training folds.")

        kf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=self.random_state)

        base_models = self._get_base_models()
        oof_layer1 = np.zeros((n_samples, len(base_models) * n_classes))
        self.layer1_models = []
        for train_idx, val_idx in kf.split(X_scaled, y):
            fold_models = []
            for model_idx, (name, model) in enumerate(base_models):
                model_clone = clone(model)
                model_clone.fit(X_scaled[train_idx], y[train_idx])
                proba = model_clone.predict_proba(X_scaled[val_idx])
                oof_layer1[val_idx, model_idx * n_classes : (model_idx + 1) * n_classes] = proba
                fold_models.append((name, model_clone))
            self.layer1_models.append(fold_models)

        self.layer1_final = []
        for name, model in base_models:
            model_clone = clone(model)
            model_clone.fit(X_scaled, y)
            self.layer1_final.append((name, model_clone))

        layer2_models = self._get_layer2_models()
        oof_layer2 = np.zeros((n_samples, len(layer2_models) * n_classes))
        self.layer2_models = []
        for train_idx, val_idx in kf.split(oof_layer1, y):
            fold_models = []
            for model_idx, (name, model) in enumerate(layer2_models):
                model_clone = clone(model)
                model_clone.fit(oof_layer1[train_idx], y[train_idx])
                proba = model_clone.predict_proba(oof_layer1[val_idx])
                oof_layer2[val_idx, model_idx * n_classes : (model_idx + 1) * n_classes] = proba
                fold_models.append((name, model_clone))
            self.layer2_models.append(fold_models)

        self.layer2_final = []
        layer1_full = self._layer1_features(X_scaled)
        for name, model in layer2_models:
            model_clone = clone(model)
            model_clone.fit(layer1_full, y)
            self.layer2_final.append((name, model_clone))

        meta_features = np.hstack([oof_layer1, oof_layer2])
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.meta_model.fit(meta_features, y)
        return self

    def _layer1_features(self, X_scaled):
        return np.hstack([model.predict_proba(X_scaled) for _, model in self.layer1_final])

    def _meta_features(self, X):
        X_scaled = self.scaler.transform(X)
        layer1 = self._layer1_features(X_scaled)
        layer2 = np.hstack([model.predict_proba(layer1) for _, model in self.layer2_final])
        return np.hstack([layer1, layer2])

    def predict(self, X):
        return self.meta_model.predict(self._meta_features(X))

    def predict_proba(self, X):
        return self.meta_model.predict_proba(self._meta_features(X))
