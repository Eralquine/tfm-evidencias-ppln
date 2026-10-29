#!/usr/bin/env python3
"""Etapa 2a: convierte cada especificacion en un prompt para el LLM generador. NO incluye la etiqueta."""
import json, sys, datetime
from gen_specs import CONTROLES
TIPOS={"politica":"una política interna","organigrama":"un extracto de organigrama con descripción de funciones","informe_formacion":"un informe de formación","procedimiento":"un procedimiento operativo","extracto_revision_accesos":"un extracto de revisión de accesos","extracto_configuracion":"un extracto de configuración","extracto_registros":"un extracto de registros","informe_escaneo":"un resumen de informe de escaneo","politica_parches":"una política de parches","resumen_pentest":"un resumen de informe de pruebas de penetración","captura_consola":"una descripción de captura de consola","registro_pruebas":"un registro de pruebas","plan":"un plan formal","informe_prueba":"un informe de prueba","informe_postincidente":"un informe post-incidente","procedimiento_sdlc":"un procedimiento de ciclo de desarrollo seguro","muestra_tickets":"una muestra de tickets","resumen_evaluacion":"un resumen de evaluación de proveedores","certificado_centro_datos":"un certificado o constancia de centro de datos","aviso_privacidad":"un aviso de privacidad","registro_actividades":"un registro de actividades de tratamiento","certificado_borrado":"un certificado de borrado","extracto_inventario":"un extracto de inventario de activos"}
LONG={"corta":"80 a 120 palabras","media":"150 a 250 palabras","larga":"300 a 450 palabras"}
EST={"formal":"tono formal y completo","resumido":"tono resumido, con frases breves","con ruido":"con ruido realista: encabezados irrelevantes, texto de relleno y un párrafo no relacionado"}
def prompt(s):
    c=next(x for x in CONTROLES if x[0]==s["control_id"]); asp=c[2]
    L=[f"Estás generando datos sintéticos, todos ficticios, para un experimento académico de clasificación de documentos (no se usarán como evidencia real). Eres un proveedor ficticio que entrega evidencia documental en una evaluación de riesgo de terceros. Escribe EN ESPAÑOL {TIPOS[s['tipo_evidencia']]} que un proveedor adjuntaría para el control «{s['control']}».",
       f"Servicio del proveedor: {s['perfil'][3:]}. La empresa se llama «{s['empresa']}». El responsable del documento es {s['responsable']}, {s['cargo']}. Usa exactamente esos nombres; inventa solo los demás nombres de sistemas y herramientas, sin repetir nombres de personas comunes.",
       f"Fecha del documento: {s['fecha_documento']} (inclúyela de forma natural en el texto)."]
    if s["etiqueta"]=="no aplicable":
        L.append("El documento menciona el tema del control de forma general o tangencial, pero el servicio no depende de él. No digas explícitamente que no aplica.")
    else:
        cub=s["aspectos_cubiertos"]
        L.append("Aspectos que el documento SÍ respalda con detalle: "+("; ".join(cub) if cub else "ninguno (habla del tema sin respaldar ningún aspecto concreto)")+".")
        no=[a for a in asp if a not in cub]
        if no: L.append("Aspectos que NO debe respaldar ni insinuar: "+"; ".join(no)+".")
        if not s["alcance_coincide"]: L.append(f"Este documento lo escribió la empresa originalmente para otro fin, sin relación con la evaluación: trata sobre {s['alcance_detalle']} (inventa su nombre concreto y escribe el documento con normalidad, indicando ese ámbito en el encabezado y en la sección de alcance). El autor no conoce ni menciona la evaluación del servicio evaluado.")
        if s["verificable"]: L.append("Incluye datos verificables: responsable con nombre y cargo, fecha de aprobación y referencia a un documento o sistema concreto.")
        else: L.append("Redáctalo de forma genérica: sin responsable nombrado, sin fecha de aprobación y sin referencias concretas.")
    L.append(f"Extensión: {LONG[s['longitud']]}. Estilo: {EST[s['estilo']]}.")
    L.append("No evalúes ni califiques el documento. Devuelve solo el texto del documento.")
    return "\n".join(L)
if __name__=="__main__":
    specs=[json.loads(l) for l in open("specs.jsonl")]
    with open("prompts_v2.jsonl","w",encoding="utf-8") as f:
        for s in specs: f.write(json.dumps({"id":s["id"],"prompt":prompt(s)},ensure_ascii=False)+"\n")
    print(len(specs),"prompts")
    for i in ("EV0226","EV0263"):
        s=next(x for x in specs if x["id"]==i); print("\n---",i,s["etiqueta"],"\n"+prompt(s))
