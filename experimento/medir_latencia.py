#!/usr/bin/env python3
"""Latencia por documento de las familias A y B (Cap. 6 sec. 6.7), medida de una en una en ESTA maquina.
  python medir_latencia.py --que A          # TF-IDF y embeddings (usa la GPU si hay), 192 docs de prueba, sin costo
  python medir_latencia.py --que B --n 24   # Claude: zero-shot, criterios y few-shot, n llamadas por modo, secuenciales (~3 USD maximo)
Salida: latencia_A.json / latencia_B.json con mediana, p95 y media en segundos."""
import os, sys, json, time, argparse, random
import numpy as np
from baseline_A import entrada, CLASES, SEMILLA

def resumen(v): v = np.array(v); return {"n": len(v), "p50_s": float(np.percentile(v, 50)), "p95_s": float(np.percentile(v, 95)), "media_s": float(v.mean())}

def lat_A(rows):
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = [r for r in rows if r["particion"] == "train"]; te = [r for r in rows if r["particion"] == "test"]; out = {}
    t0 = time.perf_counter(); v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); X = v.fit_transform([entrada(r) for r in tr])
    clf = LinearSVC(C=1, max_iter=20000, random_state=SEMILLA).fit(X, [r["etiqueta"] for r in tr]); out["tfidf_entrenamiento_s"] = time.perf_counter() - t0
    t = []
    for r in te:
        t0 = time.perf_counter(); clf.predict(v.transform([entrada(r)])); t.append(time.perf_counter() - t0)
    out["tfidf_svm_inferencia"] = resumen(t)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"; m = SentenceTransformer("intfloat/multilingual-e5-base", device=dev); m.max_seq_length = 512
        m.encode(["query: calentamiento"], normalize_embeddings=True)
        Xe = m.encode([entrada(r) for r in tr], normalize_embeddings=True); clf2 = LinearSVC(C=1, max_iter=20000, random_state=SEMILLA).fit(Xe, [r["etiqueta"] for r in tr])
        t = []
        for r in te:   # mismo troceado que baseline_A: trozos de 220 palabras con encabezado
            pal = r["texto"].split(); cab = f"query: Control: {r['control']}. Servicio: {r['perfil'][3:]}. "
            t0 = time.perf_counter()
            E = m.encode([cab + " ".join(pal[a:a + 220]) for a in range(0, max(len(pal), 1), 220)], normalize_embeddings=True)
            x = E.mean(0, keepdims=True); x /= np.linalg.norm(x); clf2.predict(x); t.append(time.perf_counter() - t0)
        out["emb_svm_inferencia"] = resumen(t); out["emb_dispositivo"] = dev
    except Exception as e: out["emb_error"] = f"{type(e).__name__}: {e}"
    return out

def lat_B(rows, n):
    import clasificar_B2 as B
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("Falta ANTHROPIC_API_KEY")
    tr = [r for r in rows if r["particion"] == "train"]; te = [r for r in rows if r["particion"] == "test"]
    rng = random.Random(5); sub = rng.sample(te, n); rng = random.Random(123)
    ej = [x for c in CLASES for x in rng.sample([r for r in tr if r["etiqueta"] == c], 2)]; rng.shuffle(ej); out = {}
    for modo in ["zeroshot", "criterios", "fewshot"]:
        t = []; ti = to = 0
        for r in sub:
            p = B.prompt(r, modo, [e for e in ej if e["id"] != r["id"]]); t0 = time.perf_counter(); txt, a, b, _ = B.llamar(p); d = time.perf_counter() - t0
            if txt: t.append(d); ti += a; to += b
            print(f"  {modo} {r['id']} {d:.2f}s", flush=True)
        out[modo] = resumen(t); out[modo]["tokens_in_medios"] = ti / max(len(t), 1); out[modo]["tokens_out_medios"] = to / max(len(t), 1)
    out["modelo"] = B.MODEL; return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--que", choices=["A", "B"], required=True); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
    rows = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]
    res = lat_A(rows) if a.que == "A" else lat_B(rows, a.n)
    json.dump(res, open(f"latencia_{a.que}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(res, indent=1))
