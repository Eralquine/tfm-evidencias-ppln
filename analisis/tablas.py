import json,collections,statistics as st
exec(open('analisis.py').read().split("res={}")[0])
res=json.load(open('res.json'))
S=['A TF-IDF+SVM','A TF-IDF+LR','A emb+SVM','A emb+LR','B zero-shot','B criterios','B few-shot','C qwen3:14b (razonando)','C gemma3:12b','C llama3.1:8b']
NM={'A TF-IDF+SVM':'TF-IDF + SVM','A TF-IDF+LR':'TF-IDF + RL','A emb+SVM':'\\emph{Embeddings} + SVM','A emb+LR':'\\emph{Embeddings} + RL',
 'B zero-shot':'Claude, \\emph{zero-shot}','B criterios':'Claude, con criterios','B few-shot':'Claude, \\emph{few-shot}',
 'C qwen3:14b (razonando)':'Qwen3 14B (razonando)','C gemma3:12b':'Gemma 3 12B','C llama3.1:8b':'Llama 3.1 8B'}
fm=lambda x,d=3: f"{x:.{d}f}".replace('.',',')
out=[]
# T1 global
t=["\\begin{table}[!htbp]\\centering\\small",
"\\caption{Resultados globales en el conjunto de prueba (n = 192). IC95: intervalo de confianza del F1 macro por \\emph{bootstrap} (2\\,000 remuestreos). AI: tasa de aceptación indebida (elaboración propia).}",
"\\label{tab:res-global}",
"\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline",
"\\textbf{Configuración} & \\textbf{F1 macro} & \\textbf{IC95} & \\textbf{Exactitud} & \\textbf{MCC} & \\textbf{Kappa} & \\textbf{AI} \\\\\\hline"]
fam=None
for s in S:
    f=s[0]
    if f!=fam:
        t.append("\\multicolumn{7}{|l|}{\\textit{%s}} \\\\\\hline"%{'A':'Familia A: representación fija y clasificador','B':'Familia B: modelo de lenguaje por API','C':'Familia C: modelos abiertos en local'}[f]); fam=f
    m=res[s]; n=NM[s]+("$^{*}$" if s.startswith('C qwen') else "")
    t.append(f"{n} & {fm(m['f1'])} & {fm(m['ci'][0])}--{fm(m['ci'][1])} & {fm(m['acc'])} & {fm(m['mcc'])} & {fm(m['kappa'])} & {fm(m['ai'])} \\\\\\hline")
t+=["\\end{tabular}","\\\\[2pt]{\\footnotesize $^{*}$ Una respuesta ilegible de 192 (n = 191 válidas).}","\\end{table}"]
out.append("\n".join(t))
# T2 F1 y recall por clase
CLS=['suficiente','parcialmente suficiente','insuficiente','no aplicable']
for key,lab,cap in [('F','tab:f1-clase','F1 por clase'),('R','tab:rec-clase','Exhaustividad (\\emph{recall}) por clase')]:
    t=["\\begin{table}[!htbp]\\centering\\footnotesize",f"\\caption{{{cap} en el conjunto de prueba (elaboración propia).}}",f"\\label{{{lab}}}",
       "\\begin{tabular}{|l|"+"c|"*len(S)+"}\\hline","\\textbf{Clase} & "+" & ".join("\\textbf{%s}"%("A1" if s=='A TF-IDF+SVM' else "A2" if s=='A TF-IDF+LR' else "A3" if s=='A emb+SVM' else "A4" if s=='A emb+LR' else "B1" if s=='B zero-shot' else "B2" if s=='B criterios' else "B3" if s=='B few-shot' else "C1" if s.startswith('C qwen') else "C2" if 'gemma' in s else "C3") for s in S)+" \\\\\\hline"]
    for i,c in enumerate(CLS):
        t.append(c.capitalize()+" & "+" & ".join(fm(res[s][key][i],2) for s in S)+" \\\\\\hline")
    if key=='F': t.append("\\textbf{Macro} & "+" & ".join(fm(res[s]['f1'],2) for s in S)+" \\\\\\hline")
    t+=["\\end{tabular}","\\\\[2pt]{\\footnotesize A1: TF-IDF + SVM; A2: TF-IDF + RL; A3: \\emph{embeddings} + SVM; A4: \\emph{embeddings} + RL; B1: \\emph{zero-shot}; B2: con criterios; B3: \\emph{few-shot}; C1: Qwen3 14B razonando; C2: Gemma 3 12B; C3: Llama 3.1 8B.}","\\end{table}"]
    out.append("\n".join(t))
# T3 subgrupos
SG=[('suficiente','Suficiente'),('par_1','Parcial, 1 aspecto'),('par_2','Parcial, 2 aspectos'),('ins_alcance','Insuficiente: alcance incorrecto'),('ins_vencida','Insuficiente: vencida'),('ins_generica','Insuficiente: genérica'),('ins_ninguna','Insuficiente: sin cobertura'),('no aplicable','No aplicable')]
t=["\\begin{table}[!htbp]\\centering\\footnotesize","\\caption{Aciertos por subgrupo de casos en el conjunto de prueba (aciertos / casos; elaboración propia). Las claves de columna son las de la tabla~\\ref{tab:f1-clase}.}","\\label{tab:subgrupos}",
   "\\begin{tabular}{|l|c|"+"c|"*len(S)+"}\\hline","\\textbf{Subgrupo} & \\textbf{n} & "+" & ".join("\\textbf{%s}"%k for k in ["A1","A2","A3","A4","B1","B2","B3","C1","C2","C3"])+" \\\\\\hline"]
for k,l in SG:
    n=[i for i in test if sub(i)==k]
    t.append(f"{l} & {len(n)} & "+" & ".join(str(sum(sys_[s].get(i)==data[i]['etiqueta'] for i in n)) for s in S)+" \\\\\\hline")
t+=["\\end{tabular}","\\end{table}"]; out.append("\n".join(t))
# T4 estilo/longitud
t=["\\begin{table}[!htbp]\\centering\\footnotesize","\\caption{Exactitud según el estilo y la longitud del documento (proporción de aciertos; elaboración propia).}","\\label{tab:estilo}",
   "\\begin{tabular}{|l|c|"+"c|"*len(S)+"}\\hline","\\textbf{Característica} & \\textbf{n} & "+" & ".join("\\textbf{%s}"%k for k in ["A1","A2","A3","A4","B1","B2","B3","C1","C2","C3"])+" \\\\\\hline"]
for key,vals in [('estilo',[('formal','Estilo formal'),('resumido','Estilo resumido'),('con ruido','Estilo con ruido')]),('longitud',[('corta','Longitud corta'),('media','Longitud media'),('larga','Longitud larga')])]:
    for v,l in vals:
        ids=[i for i in test if data[i]['meta'][key]==v]
        t.append(f"{l} & {len(ids)} & "+" & ".join(fm(sum(sys_[s].get(i)==data[i]['etiqueta'] for i in ids)/len(ids),2) for s in S)+" \\\\\\hline")
t+=["\\end{tabular}","\\end{table}"]; out.append("\n".join(t))
# T5 matrices
def mat(s,titulo):
    M=res[s]['M']; ab=['Suf.','Parc.','Insuf.','N.A.']
    r=["\\begin{minipage}[t]{0.48\\textwidth}\\centering\\footnotesize\\textbf{%s}\\\\[2pt]"%titulo,"\\begin{tabular}{|l|c|c|c|c|}\\hline"," & "+" & ".join("\\textbf{%s}"%a for a in ab)+" \\\\\\hline"]
    for i,a in enumerate(ab): r.append("\\textbf{%s} & "%a+" & ".join(str(x) for x in M[i])+" \\\\\\hline")
    r+=["\\end{tabular}\\end{minipage}"]; return "\n".join(r)
for lab,pair,cap in [('tab:mat1',('A TF-IDF+SVM','B criterios'),'TF-IDF + SVM y Claude con criterios'),('tab:mat2',('B few-shot','C qwen3:14b (razonando)'),'Claude \\emph{few-shot} y Qwen3 14B razonando')]:
    out.append("\\begin{table}[!htbp]\\centering\n\\caption{Matrices de confusión en el conjunto de prueba: %s (filas, clase real; columnas, clase predicha; Suf.: suficiente; Parc.: parcialmente suficiente; Insuf.: insuficiente; N.A.: no aplicable; elaboración propia).}\n\\label{%s}\n%s\\hfill\n%s\n\\end{table}"%(cap,lab,mat(pair[0],NM[pair[0]]),mat(pair[1],NM[pair[1]])))
# T6 McNemar
def mc(a,b,ids=test):
    x=sum(1 for i in ids if sys_[a].get(i)==data[i]['etiqueta'] and sys_[b].get(i)!=data[i]['etiqueta'] and sys_[b].get(i) in CL or (sys_[a].get(i)==data[i]['etiqueta'] and sys_[b].get(i) is None))
    return x
def mcn(a,b):
    ids=[i for i in test if sys_[a].get(i) in CL and sys_[b].get(i) in CL]
    x=sum(1 for i in ids if sys_[a][i]==data[i]['etiqueta'] and sys_[b][i]!=data[i]['etiqueta'])
    y=sum(1 for i in ids if sys_[a][i]!=data[i]['etiqueta'] and sys_[b][i]==data[i]['etiqueta'])
    n=x+y; k=min(x,y); p=min(1,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1
    return x,y,p,len(ids)
PR=[('B criterios','B few-shot'),('B criterios','B zero-shot'),('B criterios','A TF-IDF+SVM'),('B few-shot','A TF-IDF+SVM'),('B zero-shot','A TF-IDF+SVM'),('C qwen3:14b (razonando)','A TF-IDF+SVM'),('B criterios','C qwen3:14b (razonando)'),('B few-shot','C qwen3:14b (razonando)'),('A TF-IDF+SVM','A emb+SVM'),('A TF-IDF+SVM','A TF-IDF+LR'),('C gemma3:12b','B zero-shot'),('C llama3.1:8b','B zero-shot')]
t=["\\begin{table}[!htbp]\\centering\\small","\\caption{Comparaciones pareadas con la prueba exacta de McNemar. $b$: casos que acierta solo la primera configuración; $c$: casos que acierta solo la segunda (elaboración propia). No se corrigen los valores $p$ por comparaciones múltiples.}","\\label{tab:mcnemar}",
 "\\begin{tabular}{|l|l|c|c|c|}\\hline","\\textbf{Primera} & \\textbf{Segunda} & $b$ & $c$ & $p$ \\\\\\hline"]
mcres={}
for a,b in PR:
    x,y,p,n=mcn(a,b); mcres[(a,b)]=(x,y,p,n)
    ps="$<0{,}001$" if p<0.001 else fm(p,3)
    t.append(f"{NM[a]} & {NM[b]} & {x} & {y} & {ps} \\\\\\hline")
t+=["\\end{tabular}","\\end{table}"]; out.append("\n".join(t))
# T7 costos
def toks(f): 
    r=[json.loads(l) for l in open(D+f)]; return sum(x['tokens_in'] for x in r),sum(x['tokens_out'] for x in r)
P_IN,P_OUT=2,10
rows=[('B zero-shot','pred_B2_zeroshot_test.jsonl'),('B criterios','pred_B2_criterios_test.jsonl'),('B few-shot','pred_B2_fewshot_test.jsonl')]
t=["\\begin{table}[!htbp]\\centering\\small","\\caption{Tokens y costo de la Familia B sobre el conjunto de prueba. El costo supone 2 USD por millón de tokens de entrada y 10 USD por millón de salida, sin descuentos (elaboración propia).}","\\label{tab:costo-B}",
 "\\begin{tabular}{|l|r|r|r|r|}\\hline","\\textbf{Configuración} & \\textbf{Entrada} & \\textbf{Salida} & \\textbf{Costo total (USD)} & \\textbf{Por documento (USD)} \\\\\\hline"]
cost={}
for s,f in rows:
    a,b=toks(f); c=a*P_IN/1e6+b*P_OUT/1e6; cost[s]=c
    t.append(NM[s]+' & '+f'{a:,}'.replace(',','\\,')+' & '+f'{b:,}'.replace(',','\\,')+' & '+fm(c,2)+' & '+fm(c/192,4)+' \\\\\\hline')
t+=["\\end{tabular}","\\end{table}"]; out.append("\n".join(t))
open('tablas.tex','w').write("\n\n".join(out).replace('{:,}',''))
print(cost)
for k,v in mcres.items(): print(k,v)
# C stats
for f in ['pred_C_qwen3-14b-think_criterios_test.jsonl','pred_C_gemma3-12b_criterios_test.jsonl','pred_C_llama3.1-8b_criterios_test.jsonl']:
    r=[json.loads(l) for l in open(D+f)]; print(f,len(r),st.mean(x['tokens_in'] for x in r),st.mean(x['tokens_out'] for x in r),st.median(x['tokens_out'] for x in r), sum(x['seg'] for x in r)/60)
