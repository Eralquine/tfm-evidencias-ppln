#!/usr/bin/env python3
"""Etapa 2b: genera el TEXTO de cada documento con un LLM, llamando a la API REST con urllib (libreria estandar, sin SDK).
La clave se lee de la variable de entorno ANTHROPIC_API_KEY (nunca se guarda en el codigo).
Entradas: specs.jsonl (gen_specs.py) y prompts_v2.jsonl (build_prompts_v2.py).  Salida: textos.jsonl (reanuda).
Uso: python gen_text.py --pilot 2     (prueba barata)
     python gen_text.py --pilot 30
     python gen_text.py --all"""
import os, re, sys, json, argparse, random, time, urllib.request, urllib.error, concurrent.futures as cf
RECHAZO = re.compile(r"^\s*(lo siento|no puedo|lamento|disculpa|i can.t|i cannot)|no puedo (redactar|escribir|generar|ayudar)", re.I)
def es_rechazo(t): return bool(RECHAZO.search(t[:300]))
MODEL = os.environ.get("GEN_MODEL", "claude-sonnet-5-5")
OUT = "textos.jsonl"; URL = "https://api.anthropic.com/v1/messages"

def cargar():
    specs = {json.loads(l)["id"]: json.loads(l) for l in open("specs.jsonl", encoding="utf-8")}
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open("prompts_v2.jsonl", encoding="utf-8")}
    hechos = set()
    if os.path.exists(OUT):
        hechos = {j["id"] for j in map(json.loads, open(OUT, encoding="utf-8")) if "error" not in j and j.get("stop_reason") != "max_tokens" and not es_rechazo(j.get("texto", ""))}
    return specs, prompts, hechos

def pedir(pid, prompt):
    cuerpo = json.dumps({"model": MODEL, "max_tokens": 2500, "thinking": {"type": "between_tools"},
                         "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    ultimo = ""
    for intento in range(4):
        req = urllib.request.Request(URL, data=cuerpo, method="POST", headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read().decode("utf-8"))
            texto = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
            base = {"id": pid, "modelo": MODEL, "stop_reason": d.get("stop_reason"),
                    "tokens_in": d["usage"]["input_tokens"], "tokens_out": d["usage"]["output_tokens"]}
            if not texto:   # respuesta sin texto (p. ej. rechazo): se registra y NO se reintenta (cada intento cuesta)
                tipos = [b.get("type") for b in d.get("content", [])]
                return {**base, "error": f"sin texto; bloques={tipos}; stop_reason={d.get('stop_reason')}"}
            if es_rechazo(texto):   # el modelo se nego a escribir el documento: se registra, no se reintenta
                return {**base, "error": "rechazo del modelo", "texto_parcial": texto[:400]}
            if d.get("stop_reason") == "max_tokens":   # texto cortado: no sirve como evidencia
                return {**base, "error": "texto cortado por max_tokens", "texto_parcial": texto}
            return {**base, "texto": texto}
        except urllib.error.HTTPError as e:
            cuerpo_err = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 529):
                ultimo = f"HTTP {e.code}: {cuerpo_err}"
            else:   # 400, 401, 403, 404...: no se reintenta, se detiene todo
                sys.exit(f"Error de la API (HTTP {e.code}): {cuerpo_err}")
        except Exception as e:
            ultimo = f"{type(e).__name__}: {e}"
        print(f"  [{pid}] intento {intento+1} fallo -> {ultimo[:300]}", flush=True)
        time.sleep(2 ** (intento + 1))
    return {"id": pid, "error": "fallo: " + ultimo}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--ids", nargs="+"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("Falta la variable ANTHROPIC_API_KEY")
    specs, prompts, hechos = cargar()
    if a.pilot:
        rng = random.Random(7); ids = []
        for cl in ["suficiente", "parcialmente suficiente", "insuficiente", "no aplicable"]:
            lote = [i for i, s in specs.items() if s["etiqueta"] == cl]
            ids += rng.sample(lote, -(-a.pilot // 4))
        ids = ids[:a.pilot]
    elif a.ids: ids = a.ids
    elif a.all: ids = list(specs)
    else: sys.exit("Usa --pilot N, --ids EV0001 ... o --all")
    ids = [i for i in ids if i not in hechos]
    print(f"Modelo: {MODEL} | pendientes: {len(ids)}", flush=True)
    tin = tout = ok = 0
    with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda i: pedir(i, prompts[i]), ids):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if "error" not in r: ok += 1
            tin += r.get("tokens_in", 0); tout += r.get("tokens_out", 0)
            print(f"  guardado {r['id']} ({ok} ok)", flush=True)
    print(f"Listo. {ok} documentos. Tokens entrada {tin}, salida {tout}.")
if __name__ == "__main__": main()
