import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
exec(open('analisis.py').read().split("res={}")[0])
res=json.load(open('res.json'))
S=['A TF-IDF+SVM','A TF-IDF+LR','A emb+SVM','A emb+LR','B zero-shot','B criterios','B few-shot','C qwen3:14b (razonando)','C gemma3:12b','C llama3.1:8b']
L=['A1 TF-IDF + SVM','A2 TF-IDF + RL','A3 Embeddings + SVM','A4 Embeddings + RL','B1 Claude zero-shot','B2 Claude con criterios','B3 Claude few-shot','C1 Qwen3 14B (razonando)','C2 Gemma 3 12B','C3 Llama 3.1 8B']
col={'A':'#6b8fb5','B':'#c97b3a','C':'#5e9e6e'}
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(6.3,3.6))
y=np.arange(len(S))[::-1]
for yi,s in zip(y,S):
    m=res[s]; ax.barh(yi,m['f1'],color=col[s[0]],height=0.65)
    ax.plot(m['ci'],[yi,yi],color='black',lw=1.2); ax.plot([m['ci'][0]]*2,[yi-.15,yi+.15],'k',lw=1.2); ax.plot([m['ci'][1]]*2,[yi-.15,yi+.15],'k',lw=1.2)
    ax.text(1.17,yi,f"{m['f1']:.3f}".replace('.',','),va='center',ha='right',fontsize=8.5)
ax.axvline(0.78,ls='--',color='#444',lw=1); ax.text(0.785,len(S)-0.45,'meta 0,78',fontsize=8,color='#444',va='bottom')
ax.set_yticks(y); ax.set_yticklabels(L); ax.set_xlim(0,1.18); ax.set_xticks([0,.2,.4,.6,.8,1.0]); ax.set_xlabel('F1 macro (n = 192) con IC95')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig_f1.pdf'); plt.close()
# heatmap
SG=[('suficiente','Suficiente'),('par_1','Parcial, 1 aspecto'),('par_2','Parcial, 2 aspectos'),('ins_alcance','Insuf.: alcance incorrecto'),('ins_vencida','Insuf.: vencida'),('ins_generica','Insuf.: genérica'),('ins_ninguna','Insuf.: sin cobertura'),('no aplicable','No aplicable')]
M=np.zeros((len(SG),len(S)));N=[]
for r,(k,l) in enumerate(SG):
    ids=[i for i in test if sub(i)==k]; N.append(len(ids))
    for c,s in enumerate(S): M[r,c]=sum(sys_[s].get(i)==data[i]['etiqueta'] for i in ids)/len(ids)
fig,ax=plt.subplots(figsize=(6.4,3.6))
im=ax.imshow(M,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(S))); ax.set_xticklabels(['A1','A2','A3','A4','B1','B2','B3','C1','C2','C3'])
ax.set_yticks(range(len(SG))); ax.set_yticklabels([f"{l} (n={n})" for (k,l),n in zip(SG,N)])
for r in range(len(SG)):
    for c in range(len(S)):
        ax.text(c,r,f"{M[r,c]:.2f}".replace('.',','),ha='center',va='center',fontsize=7.5,color='white' if M[r,c]>0.6 else 'black')
plt.colorbar(im,ax=ax,label='Proporción de aciertos',fraction=0.04)
plt.tight_layout(); plt.savefig('fig_subgrupos.pdf'); plt.close()
print('ok')
