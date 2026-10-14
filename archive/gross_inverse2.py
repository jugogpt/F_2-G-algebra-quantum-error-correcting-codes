import sys,itertools,random,collections,pickle,time; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
import numpy as np
exec(open('gross_inverse.py').read().split("t0=time.time()")[0])
from bb_theta_inverse import constraint_matrix
gd=pickle.load(open('gross_gates2.pkl','rb')); syms=gd['syms']; folds=gd['folds']
v=np.concatenate([_vec(f1,mons),_vec(f2,mons)])
Cs=[np.array(constraint_matrix(L,s[0],s[1],s[2]),dtype=np.uint8) for s in syms]
Fs=[fold_gate_matrix(L,*f[1:]) for f in folds]
from bb_equiv import equivalence_keys
gk=equivalence_keys(f1,f2,rd)
def space(mats):
    N=GF2(np.vstack(mats)).null_space()
    return [(_elt(L,u[:D],mons),_elt(L,u[D:],mons)) for u in N]
def report(name,mats,iters=800,seed=0):
    V=space(mats); cands={}
    ng=0
    for a,b in enumerate_sparse(rd,{'basis':V},8,iters,seed,3,2):
        if len(a.mons)+len(b.mons)>8: continue
        k=fast_dimension(a,b,rd)
        if k==0: continue
        cands[(tuple(sorted(a.mons)),tuple(sorted(b.mons)))]=k
    classes={}
    for (a,b),k in cands.items():
        r=min(equivalence_keys(RingElement(L,a),RingElement(L,b),rd)); classes.setdefault(r,(a,b,k))
    isgross=any(gk & equivalence_keys(RingElement(L,a),RingElement(L,b),rd) for a,b,k in classes.values())
    print(name,"| dim",len(V),"| sparse codes",len(cands),"| classes",len(classes),"| k hist",sorted(collections.Counter(c[2] for c in classes.values()).items()),"| contains gross class:",isgross,flush=True)
    return V,classes
rng=random.Random(3)
out={}
out['all']=report("ALL 13 syms + 72 folds",Cs+Fs)
out['syms_only']=report("all 13 syms only",Cs)
out['folds_only']=report("all 72 folds only",Fs)
for t in range(6):
    si=rng.sample(range(len(Cs)),2); fi=rng.sample(range(len(Fs)),4)
    out[f'g{t}']=report(f"gamma {t}: syms {si} folds {fi}",[Cs[i] for i in si]+[Fs[i] for i in fi],seed=t)
pickle.dump({k:(len(v[0]),list(v[1].values())) for k,v in out.items()},open('gross_inverse2.pkl','wb'))
