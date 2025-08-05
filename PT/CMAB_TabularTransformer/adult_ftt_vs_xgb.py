import os
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml

from xgboost import XGBClassifier


# ===============================
# 1) Load UCI Adult (OpenML)
# ===============================
def load_adult_from_openml(test_size=0.2, seed=42):
    """
    Loads 'adult' (OpenML) as a DataFrame and returns:
    (X_train_df, X_test_df, y_train, y_test, cat_cols, num_cols)
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()

    # Standard target name in OpenML version is "class"
    y = (df["class"] == ">50K").astype(int).values
    X = df.drop(columns=["class"])

    # Clean: replace "?" with NaN, then simple impute with mode for cats / median for nums
    X = X.replace("?", np.nan)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Simple imputation
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode()[0])
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train_df, X_test_df, y_train, y_test, cat_cols, num_cols


# =========================================
# 2) FT-Transformer style tokenization
# =========================================
class NumericFeatureTokenizer(nn.Module):
    """One linear 1->d_model per numeric column (no weight sharing)."""
    def __init__(self, n_num_features: int, d_model: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_num_features)])

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: [B, F_num]
        tokens = [layer(x_num[:, i:i+1]) for i, layer in enumerate(self.proj)]  # each [B, d_model]
        return torch.stack(tokens, dim=1)  # [B, F_num, d_model]


class CategoricalFeatureTokenizer(nn.Module):
    """Embedding table per categorical column with its own cardinality."""
    def __init__(self, cardinalities: List[int], d_model: int):
        super().__init__()
        # reserve 0 as a valid id; assume already encoded in [0, card-1]
        self.emb = nn.ModuleList([nn.Embedding(card, d_model) for card in cardinalities])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [B, F_cat] (long dtype)
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb)]  # each [B, d_model]
        return torch.stack(tokens, dim=1)  # [B, F_cat, d_model]


class FTTransformerBackbone(nn.Module):
    """Attends across (numeric+categorical) feature tokens."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, F_total, d_model]
        return self.encoder(tokens)  # [B, F_total, d_model]


class TabularFTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: List[int], n_classes: int, d_model=64):
        super().__init__()
        self.has_num = (n_num > 0)
        self.has_cat = (len(cat_cardinalities) > 0)

        if self.has_num:
            self.num_tok = NumericFeatureTokenizer(n_num, d_model)
        if self.has_cat:
            self.cat_tok = CategoricalFeatureTokenizer(cat_cardinalities, d_model)

        self.backbone = FTTransformerBackbone(d_model=d_model, nhead=4, num_layers=2, dropout=0.1)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.has_num:
            toks.append(self.num_tok(x_num))         # [B, F_num, d_model]
        if self.has_cat:
            toks.append(self.cat_tok(x_cat.long()))  # [B, F_cat, d_model]

        x = torch.cat(toks, dim=1)                   # [B, F_total, d_model]
        h = self.backbone(x)                         # [B, F_total, d_model]
        pooled = h.mean(dim=1)                       # mean pool over features
        return self.classifier(pooled)               # [B, n_classes]


# =========================================
# 3) Utilities: encoders for Torch model
# =========================================
def build_torch_inputs(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: List[str], num_cols: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], StandardScaler, List[dict]]:
    """
    - Ordinal-encode categoricals (keeps per-column cardinality)
    - Standardize nums
    - Return torch tensors + metadata
    """
    # Ordinal encode categoricals per column keeping mapping (category -> id)
    cat_maps = []
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_test_cat = pd.DataFrame(index=X_test.index)
    cat_cardinalities = []

    for c in cat_cols:
        # build mapping
        cats = pd.Categorical(X_train[c]).categories.tolist()
        cat_to_id = {v: i for i, v in enumerate(cats)}
        # apply
        X_train_cat[c] = X_train[c].map(cat_to_id).fillna(0).astype(int)
        X_test_cat[c]  = X_test[c].map(cat_to_id).fillna(0).astype(int)
        # cardinality (include unseen mapped to 0—already handled)
        card = max(X_train_cat[c].max(), X_test_cat[c].max()) + 1
        cat_cardinalities.append(int(card))
        cat_maps.append({"column": c, "map": cat_to_id, "cardinality": int(card)})

    # Standardize numeric columns
    scaler = StandardScaler()
    if len(num_cols) > 0:
        X_train_num = scaler.fit_transform(X_train[num_cols].values)
        X_test_num  = scaler.transform(X_test[num_cols].values)
    else:
        X_train_num = np.zeros((len(X_train), 0), dtype=np.float32)
        X_test_num  = np.zeros((len(X_test), 0), dtype=np.float32)

    # To torch
    Xtr_num_t = torch.tensor(X_train_num, dtype=torch.float32)
    Xte_num_t = torch.tensor(X_test_num,  dtype=torch.float32)
    Xtr_cat_t = torch.tensor(X_train_cat.values, dtype=torch.long) if len(cat_cols) else torch.zeros((len(X_train),0), dtype=torch.long)
    Xte_cat_t = torch.tensor(X_test_cat.values, dtype=torch.long)  if len(cat_cols) else torch.zeros((len(X_test),0), dtype=torch.long)

    return Xtr_num_t, Xte_num_t, Xtr_cat_t, Xte_cat_t, cat_cardinalities, scaler, cat_maps


# =========================================
# 4) Train / Evaluate FT-Transformer
# =========================================
def train_ftt(
    Xtr_num: torch.Tensor, Xte_num: torch.Tensor,
    Xtr_cat: torch.Tensor, Xte_cat: torch.Tensor,
    y_train: np.ndarray, y_test: np.ndarray,
    cat_cardinalities: List[int],
    epochs=10, batch_size=1024, d_model=64, lr=1e-3, seed=42
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = TabularFTTransformer(
        n_num=Xtr_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        n_classes=2,
        d_model=d_model
    ).to(device)

    train_ds = TensorDataset(Xtr_num, Xtr_cat, torch.tensor(y_train, dtype=torch.long))
    test_ds  = TensorDataset(Xte_num, Xte_cat, torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            logits = model(xb_num, xb_cat)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb_num.size(0)

        # quick eval per epoch
        model.eval()
        with torch.no_grad():
            preds, probs = [], []
            for xb_num, xb_cat, yb in test_loader:
                out = model(xb_num.to(device), xb_cat.to(device))
                p = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
                preds.append((p >= 0.5).astype(int))
                probs.append(p)
            preds = np.concatenate(preds)
            probs = np.concatenate(probs)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"[FTT] Epoch {epoch:02d} | loss {total_loss/len(train_ds):.4f} | acc {acc:.4f} | auc {auc:.4f}")

    return model


# =========================================
# 5) XGBoost baseline (one-hot + scaler)
# =========================================
def train_xgb(
    X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
    y_train: np.ndarray, y_test: np.ndarray,
    cat_cols: List[str], num_cols: List[str], seed=42
):
    # ColumnTransformer: one-hot for categorical, standardize numeric
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    Xtr = pre.fit_transform(X_train_df)
    Xte = pre.transform(X_test_df)

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
        eval_metric="logloss",
    )
    clf.fit(Xtr, y_train)

    p = clf.predict_proba(Xte)[:, 1]
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p)
    print(f"[XGB] acc {acc:.4f} | auc {auc:.4f}")
    return clf, pre


# =========================================
# 6) Main
# =========================================
if __name__ == "__main__":
    # Load data
    X_train_df, X_test_df, y_train, y_test, cat_cols, num_cols = load_adult_from_openml()

    # Torch inputs (ordinal cats for embeddings, standardized nums)
    Xtr_num_t, Xte_num_t, Xtr_cat_t, Xte_cat_t, cat_cards, scaler, cat_maps = build_torch_inputs(
        X_train_df, X_test_df, cat_cols, num_cols
    )

    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")
    print(f"Category cardinalities: {cat_cards}")

    # Train/eval FT-Transformer
    ftt_model = train_ftt(
        Xtr_num_t, Xte_num_t, Xtr_cat_t, Xte_cat_t, y_train, y_test,
        cat_cardinalities=cat_cards,
        epochs=15, batch_size=1024, d_model=64, lr=1e-3
    )

    # Train/eval XGBoost baseline
    xgb_model, xgb_pre = train_xgb(X_train_df, X_test_df, y_train, y_test, cat_cols, num_cols)

