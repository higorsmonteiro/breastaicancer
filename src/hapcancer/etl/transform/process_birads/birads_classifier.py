from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_X_y, check_array
from joblib import dump, load
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
import hashlib
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight

from hapcancer.schemas.enums import MammogramColumns
from hapcancer.config_manager import ConfigInterface

RARE = [4,5,6]
REVIEW_FLAG = -1

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("latin")).hexdigest()

def clean_text_series(s: pd.Series) -> pd.Series:
    # -- safe “cleaning”
    return s.fillna("").astype(str).str.strip()

def tune_threshold_for_class(y_true, proba, target_class, class_to_idx):
    """
        y_true: true labels array (values in classes)
        proba: 2D array [n, K] of calibrated probs aligned to 'classes'
        target_class: int label (e.g., 4, 5, or 6)
        grid: list/array of thresholds to test
        Returns: best_threshold, (prec, rec, f1)
    """
    grid = np.linspace(0.1, 0.99, 40)
    idx = class_to_idx[target_class]
    p = proba[:, idx]
    best = (0.0, (0.0, 0.0, 0.0))  # (threshold, metrics)

    for t in grid:
        pred_positive = (p >= t).astype(int)
        true_positive = (y_true == target_class).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_positive, pred_positive, average='binary', zero_division=0
        )
        if f1 > best[1][2]:
            best = (t, (prec, rec, f1))
    return best

def predict_with_abstain(proba_row, thresholds, classes, class_to_idx, rare=RARE, review_when_uncertain=True):
    COMMON = [c for c in classes if c not in RARE]
    # Collect rare candidates above their thresholds
    candidates = []
    for c in rare:
        idx = class_to_idx[c]
        if proba_row[idx] >= thresholds[c]:
            candidates.append((proba_row[idx], c))
    if candidates:
        # pick the rare class with highest prob above threshold
        return max(candidates, key=lambda z: z[0])[1], "rare_above_threshold"

    # No rare class clears threshold:
    # Option A: fall back to {1,2,3} argmax
    # Option B (safer): if the global top is rare but below threshold, return REVIEW_FLAG
    top_idx = int(np.argmax(proba_row))
    top_class = classes[top_idx]
    if review_when_uncertain and top_class in rare:
        return REVIEW_FLAG, "rare_below_threshold"

    # else pick argmax among common classes
    # if all common classes have tiny probs, this still picks the best of them
    common_idxs = [class_to_idx[c] for c in COMMON]
    best_common_idx = common_idxs[np.argmax(proba_row[common_idxs])]
    return classes[best_common_idx], "common_argmax"

def show_top_features_per_class(clf, feature_names, classes, top_n=15):
    explain_text = []
    for i, c in enumerate(classes):
        coefs = clf.coef_[i]
        top_pos = np.argsort(coefs)[-top_n:]
        top_neg = np.argsort(coefs)[:top_n]
        explain_text.append(f"\n=== Class {c} ===")
        explain_text.append("Top positive features:")
        for j in reversed(top_pos):
            explain_text.append(f"  {feature_names[j]:<25} {coefs[j]:.3f}")
            #print(f"  {feature_names[j]:<25} {coefs[j]:.3f}")
        explain_text.append("Top negative features:")
        for j in top_neg:
            explain_text.append(f"  {feature_names[j]:<25} {coefs[j]:.3f}")
    final_explain_text = '\n'.join(explain_text)
    return final_explain_text
    

class BiradsClassifier(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)

        # -- preprocessed files (without consolidating extracted BI-RADS by RE) - texts are included
        self.preproc_birads_files = list(self.processed_birads_folder_path.glob("*single*.parquet"))
        self.preproc_birads_files += list(self.processed_birads_folder_path.glob("*multiple*.parquet"))
        # -- processed birads file (consolidated BI-RADS extracted by RE) - no texts
        self.processed_birads_file = list(self.processed_birads_folder_path.glob("processed*"))

        self.with_birads_df = None
        self.without_birads_df = None
        self.df_train = None
        self.df_infer = None
        self.df_preds = None
        self.n_classes = 7 # -> individual BI-RADS 0 to 6
        self.class_to_idx = None
        self.classes = None
        self.thresholds = None

        self.birads_clf_config = self.birads_clf_cfg["birads_classifier"]
        self.max_samples_per_class = self.birads_clf_config["max_samples_per_class"]
        self.val_size = self.birads_clf_config["val_size"]
        self.split_random_state = self.birads_clf_config["split_random_state"]
        self.tfidf_max_features = self.birads_clf_config["tfidf_max_features"]
        self.ngram_range_max = self.birads_clf_config["ngram_range_max"]
        self.clf_penalty = self.birads_clf_config["clf_penalty"]
        self.clf_solver = self.birads_clf_config["clf_solver"]

        self.pipeline = None
        self.explaining_text = None
        self.report_df = None
        self.report_df_thr = None


    def _load_processed_data(self):
        processed_df = pd.read_parquet(self.processed_birads_file[0])
        processed_df["key"] = processed_df[MammogramColumns.CD_ATENDIMENTO.value].apply(lambda x: f"{x:.0f}") + processed_df["raw_text_hash"]
        self.with_birads_df = processed_df[pd.notna(processed_df["processed_birads"])]
        self.without_birads_df = processed_df[pd.isna(processed_df["processed_birads"])]

    def _create_training_set(self, random_state: Optional[int] = 42):
        # -- create the training dataset
        total_rec = self.with_birads_df["processed_birads"].value_counts().sort_index()
        self.df_train = []
        for k in range(self.n_classes):
            temp_k = self.with_birads_df[self.with_birads_df["processed_birads"]==k].copy()
            # -- avoid repeated texts for those classes (diversify vocab)
            # -- do not do this for classes 4/5/6 because it might reduce too much number of docs.
            
            if total_rec.loc[k]<self.max_samples_per_class:
                # -- no need for oversampling
                group_k = temp_k.sample(n=total_rec.loc[k], random_state=random_state)
            else:
                group_k = temp_k.sample(n=self.max_samples_per_class, random_state=random_state)

            self.df_train.append(group_k)
        self.df_train = pd.concat(self.df_train)[["key", "processed_birads"]].drop_duplicates(subset=["key"]).copy()

    def _create_infer_set(self):
        # -- create the dataset for which we should infer the BI-RADS classes (the ones where the RE routine couldn't do it)
        # -- get the text of those records without an extracted BI-RADS
        no_birads_keys = self.without_birads_df["key"]
    
        self.df_infer, df_train_temp = [], []
        for cur_file in tqdm(self.preproc_birads_files):
            cur_df = pd.read_parquet(cur_file)
            # -- define the key
            cur_df["raw_text_hash"] = cur_df[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: sha1(x.strip()) if pd.notna(x) else np.nan)
            cur_df["key"] = cur_df[MammogramColumns.CD_ATENDIMENTO.value].apply(lambda x: f"{x:.0f}") + cur_df["raw_text_hash"]
            
            # -- for infer
            cur_df_infer = cur_df[cur_df["key"].isin(no_birads_keys)][[
                "key", MammogramColumns.CD_ATENDIMENTO.value, MammogramColumns.DT_ATENDIMENTO.value, MammogramColumns.CD_PACIENTE.value, MammogramColumns.DS_LAUDO_MEDICO.value
            ]].copy()
            self.df_infer.append(cur_df_infer)
            # -- we need the texts also for the training set
            cur_df_train = cur_df[cur_df["key"].isin(self.df_train["key"])][["key", MammogramColumns.DS_LAUDO_MEDICO.value]].copy()
            df_train_temp.append(cur_df_train)

        self.df_infer = pd.concat(self.df_infer).drop_duplicates(subset=["key"])
        df_train_temp = pd.concat(df_train_temp).drop_duplicates(subset=["key"])

        if self.df_train is None: 
            raise Exception("create training set first.")
        self.df_train = self.df_train.merge(df_train_temp, on="key", how="left")
        self.df_train = self.df_train.sort_values(by=["key", MammogramColumns.DS_LAUDO_MEDICO.value]).drop_duplicates(subset="key", keep='first')

    def _create_datasets(self, random_state: Optional[int] = 42):
        self._create_training_set(random_state=random_state)
        self._create_infer_set()

    def _fit_pipeline(self):
        
        df_train_ = self.df_train.copy()
        df_train_[MammogramColumns.DS_LAUDO_MEDICO.value] = clean_text_series(df_train_[MammogramColumns.DS_LAUDO_MEDICO.value])
        df_train_["processed_birads"] = df_train_["processed_birads"].astype(int)

        df_infer_ = self.df_infer.copy()
        df_infer_[MammogramColumns.DS_LAUDO_MEDICO.value] = clean_text_series(df_infer_[MammogramColumns.DS_LAUDO_MEDICO.value])

        X_text = df_train_[MammogramColumns.DS_LAUDO_MEDICO.value].values
        X_text_np = X_text.to_numpy(dtype=str)
        y = df_train_["processed_birads"].values

        # -- train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_text_np, y, test_size=self.val_size, random_state=self.split_random_state, stratify=y
        )

        # ======================
        # 3) Vectorizer (TF–IDF) + Linear model
        #    - Portuguese stopwords
        #    - 1-3 grams to capture phrases like "nódulo espiculado"
        # ======================
        tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words=stopwords.words('portuguese'),
            ngram_range=(1,self.ngram_range_max),
            max_features=self.tfidf_max_features,   # cap vocabulary to keep things light
            min_df=2                                # ignore singletons
        )

        base_clf = LogisticRegression(
            solver=self.clf_solver,
            max_iter=2000,
            class_weight="balanced",
            C=1.0
        )

        # We wrap the base classifier in a probability calibrator
        # Using isotonic calibration (better for strong monotonic distortions), with 3-fold internal CV
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="isotonic",
            cv=2
        )

        # -- pipeline: TFIDF + Calibrated Logistic Regression
        self.pipeline = make_pipeline(tfidf, base_clf) # -- no calibration so far (easier to explain classifications)

        '''4) Fit'''
        self.pipeline.fit(X_train, y_train)

        '''5) Evaluate on validation (argonmax prediction, no thresholds yet)'''
        y_val_proba = self.pipeline.predict_proba(X_val)  # shape: [n_val, n_classes]
        self.classes = self.pipeline.named_steps['logisticregression'].classes_
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        y_val_pred = self.classes[np.argmax(y_val_proba, axis=1)]

        print("== Plain argmax (no thresholds) on validation ==")
        print(classification_report(y_val, y_val_pred, digits=3))
        self.report_df = classification_report(y_val, y_val_pred, digits=3, output_dict=True)
        self.report_df = pd.DataFrame(self.report_df).transpose()

        ''' 6) Tune thresholds for BI-RADS 4/5/6
            Goal: prefer higher precision/recall for rare classes.
            Simple approach: per-class F1 maximization over a grid.
        '''
        self.thresholds = {}
        metrics_at_thr = {}
        for c in [4,5,6]:
            t, m = tune_threshold_for_class(y_val, y_val_proba, target_class=c, class_to_idx=self.class_to_idx)
            self.thresholds[c] = t
            metrics_at_thr[c] = m

        print("\n== Tuned self.thresholds for (4,5,6) maximizing F1 on validation ==")
        for c in [4,5,6]:
            prec, rec, f1 = metrics_at_thr[c]
            print(f"Class {c}: threshold={self.thresholds[c]:.3f}  |  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

        # Evaluate the abstention strategy on validation
        preds = []
        reasons = []
        for row in y_val_proba:
            yhat, why = predict_with_abstain(row, self.thresholds, self.classes, self.class_to_idx, review_when_uncertain=True)
            preds.append(yhat)
            reasons.append(why)

        # -- for metrics, ignore "REVIEW" rows
        mask_eval = np.array([p != REVIEW_FLAG for p in preds])
        print("\n== Evaluation with self.thresholds + abstention (ignoring 'REVIEW') ==")
        if mask_eval.any():
            print(classification_report(y_val[mask_eval], np.array(preds)[mask_eval].astype(int), digits=3))
            self.report_df_thr = classification_report(y_val[mask_eval], np.array(preds)[mask_eval].astype(int), digits=3, output_dict=True)
            self.report_df_thr = pd.DataFrame(self.report_df_thr).transpose()
        else:
            print("All validation cases were routed to REVIEW (self.thresholds too strict).")

        # Coverage: fraction auto-labeled
        coverage = mask_eval.mean()
        print(f"Coverage (auto-labeled fraction): {coverage:.3f}")

    def _infer_unlabeled(self):
        self.df_infer = self.df_infer[pd.notna(self.df_infer[MammogramColumns.DS_LAUDO_MEDICO.value])]
        proba_infer = self.pipeline.predict_proba(self.df_infer[MammogramColumns.DS_LAUDO_MEDICO.value].values)
        out_rows = []
        for key, cd_atend, patient, dt_atend, row_p in zip(self.df_infer["key"].values, 
                                                           self.df_infer[MammogramColumns.CD_ATENDIMENTO.value].values,
                                                           self.df_infer[MammogramColumns.CD_PACIENTE.value].values,
                                                           self.df_infer[MammogramColumns.DT_ATENDIMENTO.value].values, 
                                                           proba_infer):
            yhat, why = predict_with_abstain(row_p, self.thresholds, self.classes, self.class_to_idx, review_when_uncertain=True)
            record = {
                "key": key,
                MammogramColumns.CD_ATENDIMENTO.value: cd_atend, 
                MammogramColumns.CD_PACIENTE.value: patient,
                MammogramColumns.DT_ATENDIMENTO.value: dt_atend,
                "predicted_birads": yhat,
                "reason": why,
                # optional: expose probabilities for 4/5/6 for auditing
                "p4": row_p[self.class_to_idx[4]] if 4 in self.classes else np.nan,
                "p5": row_p[self.class_to_idx[5]] if 5 in self.classes else np.nan,
                "p6": row_p[self.class_to_idx[6]] if 6 in self.classes else np.nan,
            }
            out_rows.append(record)
        self.df_preds = pd.DataFrame(out_rows)

    def _explain_classes(self):
        vectorizer = self.pipeline.named_steps["tfidfvectorizer"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        clf = self.pipeline.named_steps["logisticregression"]
        self.explaining_text = show_top_features_per_class(clf, feature_names, self.classes, top_n=15)
    
    def _save(self):
        self.df_preds.to_parquet(self.processed_birads_folder_path.joinpath("infered_birads_bov.parquet"))
        with open(self.processed_birads_folder_path.joinpath("topk_explaining_features.txt"), "w", encoding="latin1") as f:
            f.write(self.explaining_text)
        self.report_df.to_csv(self.processed_birads_folder_path.joinpath("class_report_thr_naive.csv"))
        self.report_df_thr.to_csv(self.processed_birads_folder_path.joinpath("class_report_thr_optimized.csv"))

    def fit_and_infer(self, training_random_state: Optional[int] = 42):
        self._load_processed_data()
        self._create_datasets(random_state=training_random_state)
        self._fit_pipeline()
        self._explain_classes()
        self._infer_unlabeled()
        self._save()



