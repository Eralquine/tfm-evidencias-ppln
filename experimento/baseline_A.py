#!/usr/bin/env python3
"""Familia A: representacion fija + clasificador lineal (LR / SVM).
Uso:  python baseline_A.py --modo tfidf            (referencia lexica, sin GPU)
      python baseline_A.py --modo emb              (embeddings multilingues, usa GPU si hay)
Entrada: dataset.jsonl  | Salida: resultados_<modo>.json y predicciones_<modo>.csv"""
import json, argparse, csv, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (f1_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix)
CLASES = ["suficiente", "parcialmente suficiente", "insuficiente", "no aplicable"]
SEMILLA = 42

def cargar():
    return [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]

def entrada(r):  # el clasificador ve control + servicio + documento (nunca la etiqueta ni la metadata)
    return f"Control evaluado: {r['control']}. Servicio del proveedor: {r['perfil'][3:]}. Documento: {r['texto']}"

def feats_tfidf(tr, otros):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = v.fit_transform([entrada(r) for r in tr])
    return Xtr, [v.transform([entrada(r) for r in o]) for o in otros]

def feats_emb(rows_all, modelo="intfloat/multilingual-e5-base", palabras_por_trozo=220):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("Dispositivo:", dev, "| modelo:", modelo)
    m = SentenceTransformer(modelo, device=dev); m.max_seq_length = 512
    textos, dueño = [], []
    for k, r in enumerate(rows_all):   # trozos con el encabezado (control+servicio) en cada uno
        pal = r["texto"].split(); cab = f"query: Control: {r['control']}. Servicio: {r['perfil'][3:]}. "
        for a in range(0, max(len(pal), 1), palabras_por_trozo):
            textos.append(cab + " ".join(pal[a:a + palabras_por_trozo])); dueño.append(k)
    E = m.encode(textos, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    dueño = np.array(dueño); X = np.vstack([E[dueño == k].mean(0) for k in range(len(rows_all))])
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def metricas(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=CLASES, zero_division=0)
    no_suf = [i for i, t in enumerate(y) if t != "suficiente"]
    return {"macro_f1": f1_score(y, p, labels=CLASES, average="macro", zero_division=0),
            "exactitud_balanceada": balanced_accuracy_score(y, p), "mcc": matthews_corrcoef(y, p),
            "kappa": cohen_kappa_score(y, p),
            "por_clase": {c: {"P": P[i], "R": R[i], "F1": F[i]} for i, c in enumerate(CLASES)},
            # riesgo critico: aceptar como suficiente una evidencia que no lo es
            "tasa_aceptacion_indebida": float(np.mean([p[i] == "suficiente" for i in no_suf])),
            "matriz_confusion(filas=real,cols=pred)": confusion_matrix(y, p, labels=CLASES).tolist()}

def ic_bootstrap(y, p, n=2000):
    rng = np.random.default_rng(SEMILLA); y = np.array(y); p = np.array(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(f1_score(y[i], p[i], labels=CLASES, average="macro", zero_division=0))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modo", choices=["tfidf", "emb"], required=True)
    a = ap.parse_args(); rows = cargar()
    tr = [r for r in rows if r["particion"] == "train"]; va = [r for r in rows if r["particion"] == "val"]
    te = [r for r in rows if r["particion"] == "test"]; print(len(tr), len(va), len(te))
    if a.modo == "tfidf":
        Xtr, (Xva, Xte) = feats_tfidf(tr, [va, te])
        Xtv, (Xte2,) = feats_tfidf(tr + va, [te])
    else:
        X = feats_emb(rows); idx = {r["id"]: k for k, r in enumerate(rows)}
        g = lambda L: X[[idx[r["id"]] for r in L]]
        Xtr, Xva, Xte, Xtv, Xte2 = g(tr), g(va), g(te), g(tr + va), g(te)
    ytr = [r["etiqueta"] for r in tr]; yva = [r["etiqueta"] for r in va]; yte = [r["etiqueta"] for r in te]
    cands = {}
    for C in [0.1, 1, 10, 100]:
        cands[("LR", C)] = lambda C=C: LogisticRegression(C=C, max_iter=5000, random_state=SEMILLA)
        cands[("SVM", C)] = lambda C=C: LinearSVC(C=C, max_iter=20000, random_state=SEMILLA)
    val = {}
    for k, f in cands.items():
        val[k] = f1_score(yva, f().fit(Xtr, ytr).predict(Xva), labels=CLASES, average="macro", zero_division=0)
        print("validación", k, round(val[k], 4))
    salida = {"modo": a.modo, "seleccion_en_validacion": {f"{k[0]}_C{k[1]}": v for k, v in val.items()}}
    for fam in ["LR", "SVM"]:   # mejor C de cada clasificador, reentrenado con train+val
        best = max([k for k in val if k[0] == fam], key=lambda k: val[k])
        pred = cands[best]().fit(Xtv, [r["etiqueta"] for r in tr + va]).predict(Xte2)
        m = metricas(yte, list(pred)); m["C"] = best[1]; m["macro_f1_IC95_bootstrap"] = ic_bootstrap(yte, list(pred))
        salida[fam] = m
        print(f"\n{fam} (C={best[1]}) TEST macro-F1 = {m['macro_f1']:.3f}  IC95 {m['macro_f1_IC95_bootstrap'][0]:.3f}-{m['macro_f1_IC95_bootstrap'][1]:.3f}"
              f" | MCC {m['mcc']:.3f} | aceptación indebida {m['tasa_aceptacion_indebida']:.3f}")
        for c in CLASES: print(f"   {c:25s} P {m['por_clase'][c]['P']:.2f}  R {m['por_clase'][c]['R']:.2f}  F1 {m['por_clase'][c]['F1']:.2f}")
        with open(f"predicciones_{a.modo}_{fam}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["id", "control_id", "real", "predicho"])
            for r, p in zip(te, pred): w.writerow([r["id"], r["control_id"], r["etiqueta"], p])
    json.dump(salida, open(f"resultados_{a.modo}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
