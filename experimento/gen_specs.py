#!/usr/bin/env python3
"""Etapa 1 del dataset: genera ESPECIFICACIONES (sin texto) con etiqueta calculada por reglas.
Uso: python3 gen_specs.py [--n 640] [--seed 42] [--out specs.jsonl]
La etiqueta sale solo de los atributos (funcion etiquetar), nunca de un LLM."""
import json, random, argparse, collections, datetime

REF = datetime.date(2026, 9, 30)  # fecha de referencia de la evaluacion

# (id, nombre, [3 aspectos], [tipos de evidencia], regla de aplicabilidad)
CONTROLES = [
 (1,"Política de seguridad de la información",["aprobada por la dirección","comunicada al personal","revisada en el último año"],["politica"],None),
 (2,"Roles y responsabilidades de seguridad",["roles definidos","responsable nombrado","segregación de funciones"],["politica","organigrama"],None),
 (3,"Concienciación y formación",["programa anual","cobertura del personal con acceso","registro de asistencia o evaluación"],["informe_formacion"],None),
 (4,"Gestión de accesos",["aprobación previa","revisión periódica de privilegios","baja oportuna"],["procedimiento","extracto_revision_accesos"],"acceso"),
 (5,"Autenticación robusta (MFA)",["MFA en acceso remoto","MFA en cuentas privilegiadas","política de contraseñas"],["extracto_configuracion"],"remoto_priv"),
 (6,"Registro y monitoreo",["eventos definidos","retención de al menos 12 meses","revisión o alertas"],["extracto_registros","procedimiento"],"opera"),
 (7,"Gestión de vulnerabilidades",["escaneo periódico","plazos de corrección por severidad","seguimiento hasta el cierre"],["informe_escaneo","politica_parches"],"opera"),
 (8,"Pruebas de penetración",["alcance incluye el servicio","ejecutadas por un tercero independiente","hallazgos con plan de remediación"],["resumen_pentest"],"expuestos"),
 (9,"Protección contra malware",["cobertura de equipos y servidores","actualización automática","respuesta a detecciones"],["politica","captura_consola"],"equipos"),
 (10,"Copias de seguridad",["frecuencia definida","copias protegidas o cifradas","pruebas de restauración"],["procedimiento","registro_pruebas"],"datos_cliente"),
 (11,"Continuidad y recuperación ante desastres",["RTO/RPO definidos","plan aprobado","pruebas anuales con resultados"],["plan","informe_prueba"],"dependencia"),
 (12,"Gestión de incidentes",["plan con roles","plazo de notificación al cliente","lecciones aprendidas"],["plan","informe_postincidente"],None),
 (13,"Criptografía",["cifrado en reposo","cifrado en tránsito","gestión de claves"],["politica","extracto_configuracion"],"datos_cliente"),
 (14,"Desarrollo seguro",["requisitos de seguridad","revisión de código o análisis estático","ambientes separados"],["procedimiento_sdlc"],"desarrolla"),
 (15,"Gestión de cambios",["solicitud y aprobación","pruebas previas","plan de reversa"],["procedimiento","muestra_tickets"],"cambia"),
 (16,"Gestión de proveedores y subcontratistas",["evaluación previa","cláusulas de seguridad en contrato","monitoreo periódico"],["politica","resumen_evaluacion"],"subcontrata"),
 (17,"Seguridad física",["control de ingreso","registro de visitas","vigilancia"],["certificado_centro_datos","politica"],"instalaciones"),
 (18,"Protección de datos personales",["base legal y finalidad","derechos de los titulares","notificación de brechas"],["aviso_privacidad","registro_actividades"],"personales"),
 (19,"Clasificación, retención y eliminación",["esquema de clasificación","plazos de retención","eliminación con constancia"],["politica","certificado_borrado"],"conserva"),
 (20,"Inventario de activos",["inventario completo","propietarios asignados","actualización periódica"],["extracto_inventario"],None),
]
# banderas de cada perfil de servicio (determinan "no aplicable")
F = ["acceso","datos_cliente","opera","desarrolla","subcontrata","instalaciones","personales","expuestos","remoto_priv","dependencia","equipos","cambia","conserva"]
def perfil(*on): return {k:(k in on) for k in F}
PERFILES = {
 "P1 SaaS multiinquilino en nube pública": perfil("datos_cliente","opera","desarrolla","subcontrata","personales","expuestos","remoto_priv","dependencia","equipos","cambia","conserva","acceso"),
 "P2 Consultoría sin acceso a sistemas ni datos": perfil(),
 "P3 Servicios de TI gestionados": perfil("acceso","datos_cliente","opera","subcontrata","expuestos","remoto_priv","dependencia","equipos","cambia","conserva"),
 "P4 Procesamiento de datos personales (nómina)": perfil("datos_cliente","opera","subcontrata","instalaciones","personales","expuestos","remoto_priv","dependencia","equipos","cambia","conserva"),
 "P5 Desarrollo de software a la medida": perfil("acceso","desarrolla","subcontrata","remoto_priv","dependencia","equipos","cambia"),
 "P6 Infraestructura o colocation": perfil("opera","subcontrata","instalaciones","expuestos","remoto_priv","dependencia","equipos","cambia"),
}
# "acceso" de P1 = acceso a datos del cliente (aplica gestión de accesos); regla 'acceso' usa acceso OR datos_cliente
def aplica(cid, pnombre):
    regla = next(c[4] for c in CONTROLES if c[0]==cid)
    p = PERFILES[pnombre]
    if regla is None: return True
    if regla == "acceso": return p["acceso"] or p["datos_cliente"]
    return p[regla]

DESAJUSTES = ["otra entidad del grupo","otro producto o servicio distinto del evaluado","solo la oficina corporativa","otra región o sede","un entorno de pruebas"]
ESTILOS = ["formal","resumido","con ruido"]
LONGITUDES = ["corta","media","larga"]

def etiquetar(s):
    """Regla de prioridad (seccion 3 del catalogo). Unica fuente de la etiqueta."""
    if not s["aplica"]: return "no aplicable"
    if (not s["alcance_coincide"]) or s["vencida"] or (not s["verificable"]) or s["n_aspectos"]==0:
        return "insuficiente"
    if s["n_aspectos"] < 3: return "parcialmente suficiente"
    return "suficiente"

def nueva(rng, cid, pn, clase):
    c = next(x for x in CONTROLES if x[0]==cid)
    asp = c[2]
    s = dict(control_id=cid, control=c[1], perfil=pn, aplica=aplica(cid,pn),
             tipo_evidencia=rng.choice(c[3]), estilo=rng.choice(ESTILOS), longitud=rng.choice(LONGITUDES),
             alcance_coincide=True, alcance_detalle=None, vencida=False, meses_antiguedad=rng.randint(1,11),
             verificable=True, aspectos_cubiertos=list(asp), causa=None, caso_frontera=None)
    if clase=="suficiente":
        pass
    elif clase=="parcialmente suficiente":
        k = rng.choice([1,2]); s["aspectos_cubiertos"] = rng.sample(asp,k)
        if k==2 and rng.random()<0.4: s["caso_frontera"]="cobertura parcial de 2 de 3 aspectos"
    elif clase=="insuficiente":
        causa = rng.choices(["alcance","vencida","generica","ninguna"],[30,25,25,20])[0]; s["causa"]=causa
        # el resto de atributos se deja aleatorio: la precedencia decide (casos dificiles)
        s["aspectos_cubiertos"] = rng.sample(asp, rng.choice([2,3,3]))
        if causa=="alcance":
            s["alcance_coincide"]=False; s["alcance_detalle"]=rng.choice(DESAJUSTES)
            if rng.random()<0.5: s["caso_frontera"]="evidencia muy detallada sobre otro servicio"
        elif causa=="vencida":
            s["vencida"]=True; s["meses_antiguedad"]=rng.choice([13,13,14,15,18,24,30])
            if s["meses_antiguedad"]<=15 and len(s["aspectos_cubiertos"])==3: s["caso_frontera"]="evidencia completa pero vencida por pocos meses"
        elif causa=="generica":
            s["verificable"]=False
        elif causa=="ninguna":
            s["aspectos_cubiertos"]=[]
    else:  # no aplicable: el documento menciona el control sin necesitarlo
        s["aspectos_cubiertos"] = rng.sample(asp, rng.choice([0,1,2]))
        s["caso_frontera"]="no aplicable con documento que menciona el control sin necesitarlo"
        s["vencida"] = rng.random()<0.2
        if s["vencida"]: s["meses_antiguedad"]=rng.randint(13,24)
    s["n_aspectos"]=len(s["aspectos_cubiertos"])
    s["fecha_documento"]=(REF - datetime.timedelta(days=30*s["meses_antiguedad"]+rng.randint(0,20))).isoformat()
    s["etiqueta"]=etiquetar(s)
    assert s["etiqueta"]==clase, (clase, s)
    return s


NOMBRES=["Andrea","Bruno","Camila","Diego","Elena","Fabián","Gloria","Héctor","Irene","Joaquín","Karina","Leandro","Mónica","Nicolás","Olga","Pablo","Renata","Sergio","Teresa","Ulises","Valeria","Walter","Ximena","Yolanda","Zoe","Alberto","Beatriz","Cristian","Daniela","Emilio","Fernanda","Gonzalo","Helena","Ignacio","Julia","Lorenzo","Mariana","Nuria","Óscar","Paula","Rafael","Sandra","Tomás","Verónica","Adrián","Blanca","César","Dolores","Esteban","Florencia"]
APELLIDOS=["Aguirre","Barrientos","Cabrera","Duarte","Echeverría","Fonseca","Galindo","Herrera","Iglesias","Jaramillo","Keller","Lozano","Maldonado","Navarro","Ochoa","Paredes","Quiroz","Rangel","Salinas","Tapia","Urbina","Vidal","Weber","Yáñez","Zamora","Arroyo","Bustos","Cordero","Del Río","Escobar","Figueroa","Guzmán","Hidalgo","Ibáñez","Juárez","Lara","Montoya","Núñez","Orozco","Peralta","Robles","Sandoval","Trejo","Uribe","Valverde","Zárate"]
CARGOS={"gob":["Director de Seguridad de la Información","Responsable de Cumplimiento","Gerente de Riesgos Tecnológicos"],"ops":["Jefe de Operaciones de TI","Responsable de Infraestructura","Gerente de Plataforma"],"gen":["Responsable de Seguridad","Gerente de Cumplimiento","Coordinador de Seguridad Corporativa"]}
EMP_A=["Nova","Alba","Ceibal","Dorado","Estrella","Faro","Granada","Horizonte","Ítaca","Jade","Kora","Lumen","Meridiano","Nexo","Orión","Prisma","Quasar","Ribera","Sendero","Tramonto","Umbral","Vértice","Zenit","Aurora","Brisa","Cumbre","Delta","Ébano","Fénix","Glaciar"]
EMP_B=["Sistemas","Soluciones","Servicios Digitales","Consultores","Tecnologías","Datos","Cloud","Networks","Software","Infraestructura","Gestión","Analytics"]
EMP_C=["S.A.","S.A. de C.V.","S.L.","S.A.S.","Ltda."]
def asignar_nombres(specs, seed):
    r0=random.Random(f"empresas-{seed}")
    combos=[(a,b,c) for a in EMP_A for b in EMP_B for c in EMP_C]; r0.shuffle(combos)
    for s,(a,b,c) in zip(specs,combos):
        r=random.Random(f"{seed}-{s['id']}")
        s["empresa"]=f"{a} {b} {c}"
        s["responsable"]=f"{r.choice(NOMBRES)} {r.choice(APELLIDOS)} {r.choice(APELLIDOS)}"
        s["cargo"]=r.choice(CARGOS["gen"])

def split(specs, rng):
    """50/20/30 estratificado por clase; dentro de cada clase se ordena por control y se asigna
    en ciclo de 10 (5 train, 2 val, 3 test) para repartir los controles en las tres particiones."""
    patron=["train"]*5+["val"]*2+["test"]*3
    por=collections.defaultdict(list)
    for s in specs: por[s["etiqueta"]].append(s)
    for cl,lst in por.items():
        rng.shuffle(lst); lst.sort(key=lambda s:s["control_id"])
        off = rng.randrange(10)
        for i,s in enumerate(lst): s["particion"]=patron[(i+off)%10]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=640); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--out",default="specs.jsonl")
    a=ap.parse_args(); rng=random.Random(a.seed)
    por_clase=a.n//4
    pares_ok=[(c[0],p) for c in CONTROLES for p in PERFILES if aplica(c[0],p)]
    pares_na=[(c[0],p) for c in CONTROLES for p in PERFILES if not aplica(c[0],p)]
    specs=[]
    for clase in ["suficiente","parcialmente suficiente","insuficiente","no aplicable"]:
        base = pares_na if clase=="no aplicable" else pares_ok
        # ciclo barajado para cubrir pares de forma pareja antes de repetir
        cola=[]
        while len(cola)<por_clase:
            lote=base[:]; rng.shuffle(lote); cola+=lote
        for cid,pn in cola[:por_clase]: specs.append(nueva(rng,cid,pn,clase))
    rng.shuffle(specs); split(specs,rng)
    for i,s in enumerate(specs,1): s["id"]=f"EV{i:04d}"
    asignar_nombres(specs,a.seed)
    with open(a.out,"w",encoding="utf-8") as f:
        for s in specs: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    print("pares aplicables:",len(pares_ok),"| pares N/A:",len(pares_na),"| total especificaciones:",len(specs))
    print(collections.Counter(s["etiqueta"] for s in specs))
    print(collections.Counter((s["particion"],s["etiqueta"]) for s in specs))
    print("causas insuficiente:",collections.Counter(s["causa"] for s in specs if s["etiqueta"]=="insuficiente"))
    print("casos frontera:",collections.Counter(s["caso_frontera"] for s in specs if s["caso_frontera"]))
if __name__=="__main__": main()
