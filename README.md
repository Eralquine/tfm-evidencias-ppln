# Clasificación automatizada de evidencias documentales en evaluaciones de riesgo de terceros mediante PLN

Código, conjunto de datos y resultados del Trabajo Fin de Estudio (Máster Universitario en Inteligencia Artificial, UNIR), dirigido por Rafael San Miguel Carrasco. Trabajo grupal de Ernesto Alejandro Quintanilla Estrada y Mario Ramiro Vázquez Casiano.

Un proveedor responde a una evaluación de riesgo con documentos, y un analista debe decidir si cada documento es evidencia **suficiente**, **parcialmente suficiente**, **insuficiente** o **no aplicable** para un control de seguridad. Este repositorio contiene un banco de pruebas que compara tres familias de enfoques sobre los mismos 192 casos de prueba:

- **Familia A:** clasificadores entrenados (TF-IDF o *embeddings* `intfloat/multilingual-e5-base`, con regresión logística o SVM).
- **Familia B:** un modelo de lenguaje por API, en tres modos (*zero-shot*, con criterios del control y con ejemplos resueltos).
- **Familia C:** modelos abiertos ejecutados en local con Ollama (Qwen3 14B, Gemma 3 12B, Llama 3.1 8B).

## Estructura

| Carpeta | Contenido |
|---|---|
| `experimento/` | Generación del conjunto de datos (`gen_specs.py`, `build_prompts_v2.py`, `gen_text.py`), las tres familias de clasificadores, la prueba de inyección de instrucciones, la medición de latencia y el propio `dataset.jsonl`. |
| `resultados/` | Predicciones y métricas de cada configuración sobre el conjunto de prueba, y el resumen de latencia e inyección. |
| `analisis/` | Scripts que calculan las métricas, intervalos de confianza, pruebas de McNemar, tablas y figuras de la memoria. |

## El conjunto de datos

`experimento/dataset.jsonl` contiene 640 documentos **sintéticos y ficticios** en español (160 por clase), para 20 controles de seguridad y 6 perfiles de servicio. La etiqueta se calcula por reglas a partir de los atributos del documento (ámbito, vigencia, verificabilidad y número de aspectos respaldados), nunca con un modelo. La partición es fija: 320 de entrenamiento, 128 de desarrollo y 192 de prueba. Ninguna empresa, persona o sistema que aparece en los textos es real.

Campos: `id`, `texto`, `etiqueta`, `control_id`, `control`, `perfil`, `particion`, `meta` (atributos que determinan la etiqueta) y `generacion`.

## Reproducir las cifras de la memoria (no requiere claves)

```bash
pip install -r requirements.txt
cd analisis
python analisis.py     # métricas, IC95 por bootstrap, McNemar; escribe res.json
python tablas.py       # tablas en formato LaTeX
python figs.py         # figuras
```

Con las predicciones incluidas en `resultados/` se obtienen, por ejemplo, un F1 macro de 0,854 (Claude con criterios), 0,886 (con ejemplos) y 0,656 (TF-IDF más SVM) sobre los 192 casos de prueba.

## Volver a ejecutar los experimentos

Desde `experimento/`:

```bash
python baseline_A.py --modo tfidf              # Familia A, sin GPU
python baseline_A.py --modo emb                # Familia A con embeddings (usa GPU si hay)

export ANTHROPIC_API_KEY=...                   # Familia B (de pago); la clave nunca se guarda en el código
python clasificar_B2.py --modo criterios --limit 12     # prueba barata
python clasificar_B2.py --modo criterios

python clasificar_C.py --modelo qwen3:14b --modo criterios --think   # Familia C, requiere Ollama en local
python inyeccion.py --backend B --simular       # prueba de inyección (--simular no llama a la API)
python medir_latencia.py --que A
```

Para regenerar el conjunto de datos: `gen_specs.py` (especificaciones y etiquetas), luego `build_prompts_v2.py` (instrucciones para el generador) y `gen_text.py` (llama a la API; requiere clave y tiene costo). Los textos generados dependen del modelo y no son bit a bit reproducibles; por eso el conjunto publicado se considera el de referencia.

## Qué no incluye

Este repositorio no contiene claves ni credenciales. Tampoco incluye, por ahora, el material de la validación humana a ciegas (hojas de respuestas y clave de etiquetas), que se publicará cuando termine el proceso de evaluación del trabajo.

## Limitaciones

Los datos son sintéticos, cortos y de un solo generador. Los criterios entregados a los modelos son una reformulación de la regla con que se etiquetó, de modo que el modo con criterios mide en parte la aplicación de una regla conocida. Cada configuración se ejecutó una sola vez. Los detalles están en el capítulo 6 y el capítulo 7 de la memoria.

## Licencia

El código se publica con la licencia MIT (`LICENSE`). El conjunto de datos y los resultados, con Creative Commons Atribución 4.0 (`DATA_LICENSE.md`).
