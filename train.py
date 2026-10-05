from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from screening.schema import CLASS_ORDER, EXCLUDED_TARGETS, FEATURES, ROOT


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"age_years", "sex", "hemoglobin", "anemia_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {sorted(missing)}")
    frame = frame.copy()
    if (pd.to_numeric(frame["age_years"], errors="coerce") < 18).any():
        raise ValueError("Финальный проект не обучается на пациентах младше 18 лет")
    if set(frame["anemia_class"].dropna().unique()) != set(CLASS_ORDER):
        raise ValueError("Датасет должен содержать все 12 классов anemia_class из кейса")

    # Targets and post-hoc labels are explicitly excluded from feature matrix.
    allowed = [c for c in FEATURES if c in frame.columns]
    if allowed != FEATURES:
        raise ValueError(f"Не хватает признаков модели: {sorted(set(FEATURES)-set(allowed))}")
    X = frame[FEATURES].copy()
    X["sex"] = X["sex"].map({"F": 0.0, "M": 1.0})
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = frame["anemia_class"].astype(str)
    return X, y


def _profile(frame: pd.DataFrame) -> dict:
    numeric_features = [f for f in FEATURES if f not in {"sex"}]
    profile: dict = {
        "rows": int(len(frame)),
        "age_min": float(pd.to_numeric(frame["age_years"], errors="coerce").min()),
        "age_max": float(pd.to_numeric(frame["age_years"], errors="coerce").max()),
        "class_counts": frame["anemia_class"].value_counts().to_dict(),
        "features": {},
        "class_medians": {},
        "class_coverage": {},
    }
    for f in numeric_features:
        s = pd.to_numeric(frame[f], errors="coerce")
        good = s.dropna()
        if good.empty:
            profile["features"][f] = {"coverage": 0.0, "p05": None, "p50": None, "p95": None}
        else:
            profile["features"][f] = {
                "coverage": float(good.notna().sum() / len(frame)),
                "p05": float(good.quantile(0.05)),
                "p50": float(good.quantile(0.50)),
                "p95": float(good.quantile(0.95)),
            }
    for cls in CLASS_ORDER:
        sub = frame[frame["anemia_class"] == cls]
        profile["class_medians"][cls] = {}
        profile["class_coverage"][cls] = {}
        for f in numeric_features:
            s = pd.to_numeric(sub[f], errors="coerce")
            med = s.median(skipna=True)
            profile["class_medians"][cls][f] = None if pd.isna(med) else float(med)
            profile["class_coverage"][cls][f] = float(s.notna().mean())
    return profile


def train(data_path: Path, output_dir: Path, n_estimators: int = 500) -> dict:
    frame = pd.read_csv(data_path)
    X, y = _prepare(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Five-fold grouped out-of-fold validation. Repeated patient IDs and exact
    # duplicate feature profiles are forced to the same fold to reduce optimistic
    # leakage on a partially synthetic dataset.
    parent = list(range(len(frame)))
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    seen_id: dict[str, int] = {}
    seen_profile: dict[tuple, int] = {}
    for i in range(len(frame)):
        pid = str(frame.iloc[i].get("patient_id", "") or "").strip()
        vals = []
        for v in X.iloc[i].to_numpy(dtype=float):
            vals.append(None if np.isnan(v) else round(float(v), 10))
        profile = tuple(vals)
        if pid:
            if pid in seen_id:
                union(i, seen_id[pid])
            else:
                seen_id[pid] = i
        if profile in seen_profile:
            union(i, seen_profile[profile])
        else:
            seen_profile[profile] = i
    groups = np.asarray([root(i) for i in range(len(frame))])
    for cls in CLASS_ORDER:
        if len(set(groups[y.to_numpy() == cls])) < 5:
            raise ValueError(f"Для {cls} недостаточно независимых групп для 5-fold проверки")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.empty(len(frame), dtype=object)
    oof_proba = np.zeros((len(frame), len(CLASS_ORDER)), dtype=float)

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=4200 + fold,
                n_jobs=-1,
            ),
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        proba = model.predict_proba(X.iloc[va])
        oof[va] = pred
        class_to_col = {c: i for i, c in enumerate(model.classes_)}
        for j, c in enumerate(CLASS_ORDER):
            if c in class_to_col:
                oof_proba[va, j] = proba[:, class_to_col[c]]

    y_idx = np.asarray([CLASS_ORDER.index(v) for v in y])
    y_onehot = np.eye(len(CLASS_ORDER), dtype=float)[y_idx]
    multiclass_brier = float(np.mean(np.sum((oof_proba - y_onehot) ** 2, axis=1)))
    top_conf = oof_proba.max(axis=1)
    top_pred = oof_proba.argmax(axis=1)
    correct = (top_pred == y_idx).astype(float)
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (top_conf >= lo) & (top_conf < hi if hi < 1.0 else top_conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(top_conf[mask].mean()) - float(correct[mask].mean()))

    metrics = {
        "validation": "5-fold stratified-group out-of-fold on supplied Sechenov dataset; repeated IDs and exact duplicate profiles stay in one fold",
        "rows": int(len(frame)),
        "under18_rows": int((pd.to_numeric(frame["age_years"], errors="coerce") < 18).sum()),
        "accuracy": float(accuracy_score(y, oof)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "macro_f1": float(f1_score(y, oof, average="macro")),
        "multiclass_brier": multiclass_brier,
        "top_score_ece_10bins": float(ece),
        "independent_groups": int(len(set(groups))),
        "classification_report": classification_report(y, oof, labels=CLASS_ORDER, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, oof, labels=CLASS_ORDER).tolist(),
        "class_order": CLASS_ORDER,
        "note": "Внутренняя техническая валидация на предоставленном частично синтетическом датасете; не является внешней клинической валидацией.",
    }

    final_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    final_model.fit(X, y)
    bundle = {
        "model": final_model,
        "features": FEATURES,
        "classes": CLASS_ORDER,
        "schema_version": 5,
    }
    model_path = output_dir / "anemia_model.joblib"
    joblib.dump(bundle, model_path)

    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 5,
        "model_id": model_hash[:16],
        "sha256": model_hash,
        "data_sha256": data_hash,
        "sklearn_version": sklearn.__version__,
        "algorithm": "RandomForestClassifier",
        "features": FEATURES,
        "classes": CLASS_ORDER,
        "training_rows": int(len(frame)),
        "dataset_age_range": [int(frame.age_years.min()), int(frame.age_years.max())],
        "probability_note": "predict_proba используется как модельный score; он не считается клинически откалиброванной вероятностью.",
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "data_profile.json").write_text(json.dumps(_profile(frame), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metadata": metadata, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "deficiency_anemia.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.trees), ensure_ascii=False, indent=2))
