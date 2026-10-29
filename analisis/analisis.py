import json,csv,math,random,collections,sys
D='../resultados/'
data={}
for l in open('../experimento/dataset.jsonl'):
    r=json.loads(l); data[r['id']]=r
CL=['suficiente','parcialmente suficiente','insuficiente','no aplicable']
def jl(f): return {json.loads(l)['id']:json.loads(l) for l in open(D+f)}
sys_={}
for n,f in [('B zero-shot','pred_B2_zeroshot_test.jsonl'),('B criterios','pred_B2_criterios_test.jsonl'),('B few-shot','pred_B2_fewshot_test.jsonl'),
            ('C qwen3:14b (razonando)','pred_C_qwen3-14b-think_criterios_test.jsonl'),('C gemma3:12b','pred_C_gemma3-12b_criterios_test.jsonl'),('C llama3.1:8b','pred_C_llama3.1-8b_criterios_test.jsonl')]:
    sys_[n]={k:v['etiqueta_pred'] for k,v in jl(f).items()}
for n,f in [('A TF-IDF+SVM','predicciones_tfidf_SVM.csv'),('A TF-IDF+LR','predicciones_tfidf_LR.csv'),('A emb+SVM','predicciones_emb_SVM.csv'),('A emb+LR','predicciones_emb_LR.csv')]:
    sys_[n]={r['id']:r['predicho'] for r in csv.DictReader(open(D+f))}
test=[k for k,v in data.items() if v['particion']=='test']
def met(pred,ids=test):
    y=[data[i]['etiqueta'] for i in ids if pred.get(i) in CL]; p=[pred[i] for i in ids if pred.get(i) in CL]
    n=len(y); M=[[0]*4 for _ in range(4)]
    for a,b in zip(y,p): M[CL.index(a)][CL.index(b)]+=1
    f1=[];P=[];R=[]
    for c in range(4):
        tp=M[c][c]; fp=sum(M[r][c] for r in range(4))-tp; fn=sum(M[c])-tp
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        P.append(pr);R.append(rc);f1.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    po=sum(M[i][i] for i in range(4))/n
    pe=sum(sum(M[i])*sum(M[r][i] for r in range(4)) for i in range(4))/n**2
    kappa=(po-pe)/(1-pe)
    # MCC multiclass
    t=[sum(M[i]) for i in range(4)]; pk=[sum(M[r][i] for r in range(4)) for i in range(4)]; c=sum(M[i][i] for i in range(4))
    num=c*n-sum(t[i]*pk[i] for i in range(4)); den=math.sqrt(n*n-sum(x*x for x in pk))*math.sqrt(n*n-sum(x*x for x in t))
    noS=[(a,b) for a,b in zip(y,p) if a!='suficiente']
    ai=sum(1 for a,b in noS if b=='suficiente')/len(noS)
    return dict(n=n,f1=sum(f1)/4,P=P,R=R,F=f1,acc=po,mcc=num/den,kappa=kappa,ai=ai,M=M)
def boot(pred,B=2000,seed=1):
    rnd=random.Random(seed); out=[]
    for _ in range(B):
        ids=[rnd.choice(test) for _ in test]; 
        # resample with duplicates
        y=[data[i]['etiqueta'] for i in ids if pred.get(i) in CL]; p=[pred[i] for i in ids if pred.get(i) in CL]
        f=[]
        for c in CL:
            tp=sum(1 for a,b in zip(y,p) if a==c and b==c); fp=sum(1 for a,b in zip(y,p) if a!=c and b==c); fn=sum(1 for a,b in zip(y,p) if a==c and b!=c)
            pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        out.append(sum(f)/4)
    out.sort(); return out[int(.025*B)],out[int(.975*B)]
def sub(i):
    m=data[i]['meta']; e=data[i]['etiqueta']
    if e=='insuficiente': return 'ins_'+m['causa']
    if e=='parcialmente suficiente': return 'par_%d'%m['n_aspectos']
    return e
def subacc(pred):
    c=collections.defaultdict(lambda:[0,0])
    for i in test:
        s=sub(i); c[s][1]+=1; c[s][0]+= (pred.get(i)==data[i]['etiqueta'])
    return dict(c)
def mcnemar(a,b):
    b01=sum(1 for i in test if a.get(i)==data[i]['etiqueta'] and b.get(i)!=data[i]['etiqueta'])
    b10=sum(1 for i in test if a.get(i)!=data[i]['etiqueta'] and b.get(i)==data[i]['etiqueta'])
    n=b01+b10
    if n==0: return b01,b10,1.0
    k=min(b01,b10); p=sum(math.comb(n,j) for j in range(0,k+1))/2**n*2
    return b01,b10,min(1,p)
res={}
for n,p in sys_.items():
    m=met(p); m['ci']=boot(p); m['sub']=subacc(p); res[n]=m
    print(f"{n:28s} F1 {m['f1']:.3f} [{m['ci'][0]:.3f},{m['ci'][1]:.3f}] acc {m['acc']:.3f} mcc {m['mcc']:.3f} k {m['kappa']:.3f} AI {m['ai']:.3f} n {m['n']}")
json.dump(res,open('res.json','w'),ensure_ascii=False,indent=1)
pairs=[('B criterios','B few-shot'),('B criterios','B zero-shot'),('B criterios','A TF-IDF+SVM'),('B few-shot','A TF-IDF+SVM'),('C qwen3:14b (razonando)','A TF-IDF+SVM'),('B criterios','C qwen3:14b (razonando)'),('B zero-shot','A TF-IDF+SVM'),('A TF-IDF+SVM','A emb+SVM'),('C qwen3:14b (razonando)','B zero-shot'),('C gemma3:12b','B zero-shot'),('C llama3.1:8b','B zero-shot')]
for a,b in pairs:
    x,y,p=mcnemar(sys_[a],sys_[b]); print(a,'vs',b,x,y,round(p,4))
