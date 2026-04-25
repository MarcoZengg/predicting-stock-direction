import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer


@dataclass
class Config:
    input_csv: str
    output_csv: str
    model_name: str
    batch_size: int
    max_length: int
    train_baseline: bool


def build_text_column(df: pd.DataFrame) -> pd.Series:
    text_cols = ["title", "summary", "category_within_source"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    combined = (
        "Title: "
        + df["title"].fillna("").astype(str)
        + " [SEP] Summary: "
        + df["summary"].fillna("").astype(str)
        + " [SEP] Category: "
        + df["category_within_source"].fillna("").astype(str)
    )
    return combined


def mean_pool(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(texts: List[str], model_name: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_text = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def append_embedding_columns(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])],
        index=df.index,
    )
    return pd.concat([df, emb_df], axis=1)


def train_prototype_classifier(feature_df: pd.DataFrame) -> None:
    if "ticker_sentiment_label" not in feature_df.columns:
        print("Skipped baseline model: ticker_sentiment_label column not found.")
        return

    labeled = feature_df.dropna(subset=["ticker_sentiment_label"])
    if labeled.empty:
        print("Skipped baseline model: no non-null labels found.")
        return

    emb_cols = [c for c in labeled.columns if c.startswith("text_emb_")]
    if not emb_cols:
        print("Skipped baseline model: embedding columns missing.")
        return

    X = labeled[emb_cols].values
    y_raw = labeled["ticker_sentiment_label"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        print("Skipped baseline model: at least two label classes are required.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\n=== Prototype Baseline (Logistic Regression on Transformer Features) ===")
    print(classification_report(y_test, preds, target_names=le.classes_))


def parse_args() -> Config:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Create transformer text features from news_sentiment.csv"
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join(project_root, "data", "news_data", "news_sentiment.csv"),
        help="Path to input news sentiment CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            project_root,
            "data",
            "news_data",
            "news_sentiment_transformer_features.csv",
        ),
        help="Path to output CSV with embedding features",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transformer encoding",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length",
    )
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Train a small baseline classifier using generated embeddings",
    )

    args = parser.parse_args()
    return Config(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_baseline=args.train_baseline,
    )


def main() -> None:
    cfg = parse_args()

    if not os.path.exists(cfg.input_csv):
        raise FileNotFoundError(f"Input file not found: {cfg.input_csv}")

    df = pd.read_csv(cfg.input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty. Please fetch news data first.")

    combined_text = build_text_column(df)
    embeddings = encode_texts(
        combined_text.tolist(),
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    feature_df = append_embedding_columns(df, embeddings)
    feature_df.to_csv(cfg.output_csv, index=False)

    print(f"Loaded {len(df)} rows from {cfg.input_csv}")
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"Saved feature table to: {cfg.output_csv}")

    if cfg.train_baseline:
        train_prototype_classifier(feature_df)


if __name__ == "__main__":
    main()