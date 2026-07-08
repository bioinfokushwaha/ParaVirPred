#!/usr/bin/env python3
"""
ParaVirPred — Offline Protein Virulence Prediction Pipeline
Supports both classical ML models (.pkl, e.g. SVM via joblib) and DNN
models (.pth, PyTorch), across all six embedding variants:

    ESM2_only, ProtT5_only, ESM2_Physchem, ProtT5_Physchem,
    ESM2_ProtT5, ESM2_ProtT5_Physchem

The required variant is auto-detected from the model filename, so only
the embeddings actually needed are computed (faster, and avoids
feature-dimension mismatches against the model's scaler).

Usage:
    python3 predict_virulence.py -i sequences.fasta -m esm2_only_dnn.pth -o output.csv
    python3 predict_virulence.py -i sequences.fasta -m prott5_physchem_svm.pkl -o output.xlsx
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

ESM2_DIM = 1280
PROTT5_DIM = 1024
PHYSCHEM_DIM = 9

# ----------------------------------------------------------------------
# Lazy imports
# ----------------------------------------------------------------------

def _lazy_import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _lazy_import_transformers():
    from transformers import T5Tokenizer, T5EncoderModel
    return T5Tokenizer, T5EncoderModel


def _lazy_import_esm():
    import esm
    return esm


def _lazy_import_biopython():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    return ProteinAnalysis


# ----------------------------------------------------------------------
# Model variant detection from filename
# ----------------------------------------------------------------------

VARIANTS = {
    # variant_name : (uses_esm2, uses_prott5, uses_physchem, expected_dim)
    "esm2_prott5_physchem": (True, True, True, ESM2_DIM + PROTT5_DIM + PHYSCHEM_DIM),
    "esm2_prott5":          (True, True, False, ESM2_DIM + PROTT5_DIM),
    "esm2_physchem":        (True, False, True, ESM2_DIM + PHYSCHEM_DIM),
    "prott5_physchem":      (False, True, True, PROTT5_DIM + PHYSCHEM_DIM),
    "esm2_only":            (True, False, False, ESM2_DIM),
    "prott5_only":          (False, True, False, PROTT5_DIM),
}

# Order matters: check the longest/most-specific patterns first
VARIANT_ORDER = [
    "esm2_prott5_physchem",
    "esm2_prott5",
    "esm2_physchem",
    "prott5_physchem",
    "esm2_only",
    "prott5_only",
]


def detect_variant(model_path):
    name = os.path.basename(model_path).lower()
    name_norm = re.sub(r"[^a-z0-9]+", "_", name)

    for variant in VARIANT_ORDER:
        if variant in name_norm:
            return variant, VARIANTS[variant]

    raise ValueError(
        f"Could not detect embedding variant from model filename '{model_path}'. "
        f"Expected the filename to contain one of: {', '.join(VARIANT_ORDER)}"
    )


# ----------------------------------------------------------------------
# FASTA parsing
# ----------------------------------------------------------------------

def parse_fasta(fasta_path):
    records = []
    header, seq_chunks = None, []
    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line.upper().replace("*", ""))
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")
    return records


# ----------------------------------------------------------------------
# Physicochemical features (9 features, ProtParam-based)
# Order: MW, pI, aromaticity, instability, gravy, helix, turn, sheet, charge_at_pH7
# ----------------------------------------------------------------------

def compute_physchem_features(sequence):
    ProteinAnalysis = _lazy_import_biopython()
    clean_seq = "".join(ch for ch in sequence if ch.isalpha())
    clean_seq = "".join(ch if ch in "ACDEFGHIKLMNPQRSTVWY" else "A" for ch in clean_seq)
    analysis = ProteinAnalysis(clean_seq)

    mw = analysis.molecular_weight()
    pi = analysis.isoelectric_point()
    aromaticity = analysis.aromaticity()
    instability = analysis.instability_index()
    gravy = analysis.gravy()
    helix, turn, sheet = analysis.secondary_structure_fraction()
    charge_at_ph7 = analysis.charge_at_pH(7.0)

    return np.array([mw, pi, aromaticity, instability, gravy,
                      helix, turn, sheet, charge_at_ph7], dtype=np.float32)


# ----------------------------------------------------------------------
# ProtT5 embeddings (1024-dim, mean-pooled)
# ----------------------------------------------------------------------

_PROTT5_MODEL = None
_PROTT5_TOKENIZER = None


def get_prott5_model(device):
    global _PROTT5_MODEL, _PROTT5_TOKENIZER
    if _PROTT5_MODEL is None:
        T5Tokenizer, T5EncoderModel = _lazy_import_transformers()
        print("Loading ProtT5 (Rostlab/prot_t5_xl_uniref50)... this can take a while the first time.")
        _PROTT5_TOKENIZER = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
        )
        _PROTT5_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        _PROTT5_MODEL = _PROTT5_MODEL.to(device).eval()
    return _PROTT5_MODEL, _PROTT5_TOKENIZER


def compute_prott5_embedding(sequence, device):
    torch, _ = _lazy_import_torch()
    model, tokenizer = get_prott5_model(device)

    spaced_seq = " ".join(list(sequence))
    spaced_seq = spaced_seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")

    ids = tokenizer(spaced_seq, return_tensors="pt", padding=True)
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    seq_len = (attention_mask[0] == 1).sum().item() - 1
    per_residue = embedding_repr.last_hidden_state[0, :seq_len]
    return per_residue.mean(dim=0).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# ESM2 embeddings (1280-dim, mean-pooled, esm2_t33_650M_UR50D)
# ----------------------------------------------------------------------

_ESM2_MODEL = None
_ESM2_ALPHABET = None
_ESM2_BATCH_CONVERTER = None


def get_esm2_model(device):
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_BATCH_CONVERTER
    if _ESM2_MODEL is None:
        esm = _lazy_import_esm()
        print("Loading ESM2 (esm2_t33_650M_UR50D)... this can take a while the first time.")
        _ESM2_MODEL, _ESM2_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
        _ESM2_BATCH_CONVERTER = _ESM2_ALPHABET.get_batch_converter()
        _ESM2_MODEL = _ESM2_MODEL.to(device).eval()
    return _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_BATCH_CONVERTER


def compute_esm2_embedding(seq_id, sequence, device):
    torch, _ = _lazy_import_torch()
    model, alphabet, batch_converter = get_esm2_model(device)

    # ESM2 has a practical context limit; truncate very long sequences
    trimmed_seq = sequence[:1022]

    _, _, batch_tokens = batch_converter([(seq_id, trimmed_seq)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_repr = results["representations"][33][0]

    # mean-pool over real residues (exclude BOS/EOS tokens)
    seq_len = len(trimmed_seq)
    mean_pooled = token_repr[1:seq_len + 1].mean(dim=0).cpu().numpy()
    return mean_pooled.astype(np.float32)


# ----------------------------------------------------------------------
# Feature extraction — only computes what the detected variant needs
# ----------------------------------------------------------------------

def extract_features(fasta_path, uses_esm2, uses_prott5, uses_physchem, device="cpu"):
    parts = []
    if uses_esm2:
        parts.append("ESM2")
    if uses_prott5:
        parts.append("ProtT5")
    if uses_physchem:
        parts.append("Physchem")
    print(f"Extracting features: {' + '.join(parts)}")

    records = parse_fasta(fasta_path)
    protein_ids = []
    feature_rows = []

    for i, (seq_id, seq) in enumerate(records, 1):
        print(f"  [{i}/{len(records)}] {seq_id} ({len(seq)} aa)")
        chunks = []
        if uses_esm2:
            chunks.append(compute_esm2_embedding(seq_id, seq, device))
        if uses_prott5:
            chunks.append(compute_prott5_embedding(seq, device))
        if uses_physchem:
            chunks.append(compute_physchem_features(seq))

        feature_rows.append(np.concatenate(chunks))
        protein_ids.append(seq_id)

    features = np.vstack(feature_rows)
    return protein_ids, features


# ----------------------------------------------------------------------
# PyTorch DNN architecture (dynamically rebuilt from checkpoint shapes)
# Standard pattern: Linear -> BatchNorm1d -> ReLU -> Dropout, repeated,
# ending in Linear(.,1). Works for any input_dim/hidden sizes that
# follow this pattern, since dims are read from the state_dict itself.
# ----------------------------------------------------------------------

def build_dnn_from_state_dict(nn, state_dict):
    # Collect Linear layer indices in order, e.g. network.0, network.4, network.8
    linear_indices = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("network.") and k.endswith(".weight")
         and state_dict[k].dim() == 2}
    )

    layers = []
    for pos, idx in enumerate(linear_indices):
        w = state_dict[f"network.{idx}.weight"]
        out_dim, in_dim = w.shape
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = (pos == len(linear_indices) - 1)
        if not is_last:
            bn_idx = idx + 1
            if f"network.{bn_idx}.running_mean" in state_dict:
                bn_dim = state_dict[f"network.{bn_idx}.running_mean"].shape[0]
                layers.append(nn.BatchNorm1d(bn_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

    return nn.Sequential(*layers)


def load_pth_model(model_path):
    torch, nn = _lazy_import_torch()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        scaler = checkpoint.get("scaler", None)
    else:
        state_dict = checkpoint
        scaler = None

    network = build_dnn_from_state_dict(nn, state_dict)

    class DNN(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.network = net

        def forward(self, x):
            return self.network(x)

    dnn = DNN(network)
    dnn.load_state_dict(state_dict)
    dnn.eval()
    return dnn, scaler


def predict_with_pth(model_path, features):
    torch, nn = _lazy_import_torch()
    dnn, scaler = load_pth_model(model_path)

    X = features
    if scaler is not None:
        X = scaler.transform(X)

    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        logits = dnn(x_tensor).squeeze(-1)
        probabilities = torch.sigmoid(logits).numpy()

    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


# ----------------------------------------------------------------------
# Classical (.pkl) model prediction
# ----------------------------------------------------------------------

def predict_with_pkl(model_path, features):
    loaded = joblib.load(model_path)

    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, scaler = loaded
    else:
        model, scaler = loaded, None

    X = features
    if scaler is not None:
        X = scaler.transform(X)

    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probabilities = 1 / (1 + np.exp(-scores))
    else:
        probabilities = predictions.astype(float)

    return predictions, probabilities


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline Protein Virulence Prediction Pipeline")
    parser.add_argument("-i", "--input", required=True, dest="input", help="Path to input FASTA file")
    parser.add_argument("-m", "--model", required=True, dest="model", help="Path to model file (.pkl or .pth)")
    parser.add_argument("-o", "--output", default="predictions_output.csv", dest="output",
                         help="Output path (.csv or .xlsx)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input FASTA file {args.input} not found.")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)

    ext = os.path.splitext(args.model)[1].lower()
    if ext not in (".pkl", ".pth"):
        print(f"Error: Unsupported model file type '{ext}'. Use .pkl or .pth.")
        sys.exit(1)

    variant_name, (uses_esm2, uses_prott5, uses_physchem, expected_dim) = detect_variant(args.model)
    print(f"Detected model variant: {variant_name}  (expects {expected_dim}-dim features)")

    device = "cpu"
    if uses_esm2 or uses_prott5:
        torch, _ = _lazy_import_torch()
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    protein_ids, features = extract_features(
        args.input, uses_esm2, uses_prott5, uses_physchem, device=device
    )

    if features.shape[1] != expected_dim:
        print(f"Error: extracted {features.shape[1]} features but variant '{variant_name}' "
              f"expects {expected_dim}. Check the embedding/feature logic.")
        sys.exit(1)

    print(f"Running inference with {ext} model: {args.model}")
    if ext == ".pkl":
        predictions, probabilities = predict_with_pkl(args.model, features)
    else:
        predictions, probabilities = predict_with_pth(args.model, features)

    labels = ["Virulent" if p == 1 else "Non-Virulent" for p in predictions]
    results = pd.DataFrame({
        "Protein_ID": protein_ids,
        "Prediction": labels,
        "Confidence_Probability": probabilities
    })

    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext == ".xlsx":
        results.to_excel(args.output, index=False)
    else:
        results.to_csv(args.output, index=False)

    n_virulent = labels.count("Virulent")
    n_non_virulent = labels.count("Non-Virulent")
    print(f"Inferences saved successfully to: {args.output}")
    print(f"Total proteins: {len(labels)}")
    print(f"Virulent: {n_virulent}")
    print(f"Non-Virulent: {n_non_virulent}")


if __name__ == "__main__":
    main()
