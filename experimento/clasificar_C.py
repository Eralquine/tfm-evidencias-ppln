#!/usr/bin/env python3
"""Familia C: modelos abiertos locales via Ollama (http://localhost:11434). Sin API de pago.
Reutiliza prompts y parseo de clasificar_B2.py (mismos prompts => comparable con la familia B).
Uso:  python clasificar_C.py --modelo qwen3:14b --modo criterios --limit 12     (prueba)
      python clasificar_C.py --modelo qwen3:14b --modo criterios                 (192 de prueba)
Salida: pred_C_<modelo>_<modo>_<split>.jsonl (reanuda) + resultados_C_...json (incluye latencia p50/p95)."""
import os, sys, json, argparse, random, time, re, urllib.request, urllib.error
import numpy as np
from clasificar_B2 import prompt, parsear, FORMATO
FORMATO_COT = ("Razona paso a paso dentro del campo \"analisis\" (máximo 120 palabras): (1) ¿el control aplica al servicio? (2) ¿el alcance del documento corresponde al servicio evaluado? "
  "(3) calcula la antigüedad del documento frente a la fecha de evaluación (máximo 12 meses) (4) ¿es verificable (responsable con cargo, fecha de aprobación, referencia concreta)? (5) ¿cuántos de los 3 aspectos exigidos respalda realmente? "
  "Después da la etiqueta. Responde SOLO con un objeto JSON con este orden de campos: {\"analisis\": \"...\", \"etiqueta\": \"<una de las cuatro clases>\", \"motivo\": \"<máximo 25 palabras>\"}")
from baseline_A import metricas, ic_bootstrap, CLASES
HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")

def llamar(modelo, p, ctx, think=False, seed=42, npred=None):
    cuerpo = {"model": modelo, "stream": False, "format": "json", "think": think, "keep_alive": "30m",
              "messages": [{"role": "user", "content": p}],
              "options": {"temperature": 0, "seed": seed, "num_ctx": ctx, "num_predict": npred or (6000 if think else 300)}}
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(cuerpo).encode(), method="POST", headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        cu = e.read().decode("utf-8", "replace")[:300]
        if "think" in cu.lower():   # modelo sin soporte de 'think': reintento sin ese campo
            cuerpo.pop("think"); req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(cuerpo).encode(), method="POST", headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r: d = json.loads(r.read().decode())
        else: sys.exit(f"Error de Ollama (HTTP {e.code}): {cu}")
    except urllib.error.URLError as e:
        sys.exit(f"No se pudo conectar con Ollama en {HOST}: {e}. ¿Está abierto Ollama?")
    seg = time.perf_counter() - t0
    return d.get("message", {}).get("content", ""), d.get("prompt_eval_count", 0), d.get("eval_count", 0), seg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modelo", required=True)
    ap.add_argument("--modo", choices=["zeroshot", "criterios", "fewshot"], required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=0); ap.add_argument("--think", action="store_true"); ap.add_argument("--cot", action="store_true", help="razonamiento escrito en el JSON (para modelos sin modo thinking)"); ap.add_argument("--simular", action="store_true"); a = ap.parse_args()
    ctx = a.ctx or (16384 if a.modo == "fewshot" else 8192)
    if a.think: ctx += 4096
    rows = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]
    tr = [r for r in rows if r["particion"] == "train"]; obj = [r for r in rows if r["particion"] == a.split]
    if a.limit:
        rng = random.Random(7); obj = [x for c in CLASES for x in rng.sample([r for r in obj if r["etiqueta"] == c], -(-a.limit // 4))][:a.limit]
    rng = random.Random(123); ejemplos = [x for c in CLASES for x in rng.sample([r for r in tr if r["etiqueta"] == c], 2)]; rng.shuffle(ejemplos)
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", a.modelo) + ("-think" if a.think else "") + ("-cot" if a.cot else "") + ("_sim" if a.simular else "")
    out = f"pred_C_{tag}_{a.modo}_{a.split}.jsonl"; hechos = {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            j = json.loads(l)
            if j.get("etiqueta_pred") in CLASES: hechos[j["id"]] = j
    pend = [r for r in obj if r["id"] not in hechos]; print(f"Modelo {a.modelo} | modo {a.modo} | {a.split} | ctx {ctx} | pendientes {len(pend)} de {len(obj)}", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for k, r in enumerate(pend, 1):
            p = prompt(r, a.modo, [e for e in ejemplos if e["id"] != r["id"]])
            if a.cot: p = p.replace(FORMATO, FORMATO_COT)
            if a.simular: txt, ti, to, seg = json.dumps({"etiqueta": random.Random(r["id"]).choice(CLASES), "motivo": "sim"}), len(p)//3, 20, 1.0
            else: txt, ti, to, seg = llamar(a.modelo, p, ctx, a.think, npred=(700 if a.cot else None))
            e, mot = parsear(txt); j = {"id": r["id"], "etiqueta_pred": e, "motivo": mot, "tokens_in": ti, "tokens_out": to, "seg": round(seg, 2), "modelo": a.modelo}
            f.write(json.dumps(j, ensure_ascii=False) + "\n"); f.flush(); hechos[r["id"]] = j if e else hechos.get(r["id"], j)
            print(f"  [{k}/{len(pend)}] {r['id']} -> {e}  ({seg:.1f}s)", flush=True)
    por = {r["id"]: r for r in obj}; ids = [i for i in por if i in hechos and hechos[i].get("etiqueta_pred") in CLASES]
    inval = len(obj) - len(ids)
    if not ids: print("Sin respuestas válidas."); return
    y = [por[i]["etiqueta"] for i in ids]; p = [hechos[i]["etiqueta_pred"] for i in ids]
    m = metricas(y, p); ic = ic_bootstrap(y, p); lat_all = np.array([hechos[i]["seg"] for i in ids]); lat = lat_all[1:] if len(lat_all) > 5 else lat_all  # se excluye la 1.ª llamada (carga del modelo en GPU)
    print(f"\n{a.modelo} {a.modo} ({a.split}, n={len(ids)}, sin respuesta válida: {inval}) macro-F1 = {m['macro_f1']:.3f}  IC95 {ic[0]:.3f}-{ic[1]:.3f} | MCC {m['mcc']:.3f} | aceptación indebida {m['tasa_aceptacion_indebida']:.3f}")
    for c in CLASES: print(f"   {c:25s} P {m['por_clase'][c]['P']:.2f}  R {m['por_clase'][c]['R']:.2f}  F1 {m['por_clase'][c]['F1']:.2f}")
    print(f"Latencia por documento: p50 {np.percentile(lat,50):.1f}s  p95 {np.percentile(lat,95):.1f}s | total {lat.sum()/60:.1f} min | tokens entrada medios {np.mean([hechos[i]['tokens_in'] for i in ids]):.0f}")
    m.update({"IC95": ic, "n": len(ids), "sin_respuesta_valida": inval, "modelo": a.modelo, "lat_p50_s": float(np.percentile(lat,50)), "lat_p95_s": float(np.percentile(lat,95))})
    json.dump(m, open(f"resultados_C_{tag}_{a.modo}_{a.split}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if __name__ == "__main__": main()
