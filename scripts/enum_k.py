# usage: enum_k.py "(6,6)" ktarget weights out.pkl ; enumerates (f1,f2) classes with exact k from fast_dimension
import sys,time,pickle,itertools; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_sweep import fast_dimension, _mons
from bb_equiv import _auts, _act, _norm, equivalence_keys
L=eval(sys.argv[1]); kt=set(eval(sys.argv[2])); W=eval(sys.argv[3]); out=sys.argv[4]
rd=ring_setup(L); mons=_mons(L); zero=tuple(0 for _ in L); others=[m for m in mons if m!=zero]
auts=_auts(rd); print("ring",L,"|Aut|",len(auts),flush=True)
def fkey(ms):
    return min(_norm([_act(p,m,L) for m in ms],L) for p in auts)
t=time.time(); reps={}
for w in set(w for pr in W for w in pr):
    for rest in itertools.combinations(others,w-1):
        ms=(zero,)+rest; kk=fkey(ms)
        reps.setdefault(w,{}).setdefault(kk,ms)
print({w:len(v) for w,v in reps.items()},"f1 classes",round(time.time()-t),"s",flush=True)
seen=set(); res=[]; kdist={}
for w1,w2 in W:
    f2s=[(zero,)+r for r in itertools.combinations(others,w2-1)]
    for kk,ms1 in reps[w1].items():
        f1=RingElement(L,ms1)
        for ms2 in f2s:
            f2=RingElement(L,ms2)
            k=fast_dimension(f1,f2,rd); kdist[k]=kdist.get(k,0)+1
            if k not in kt: continue
            ks=equivalence_keys(f1,f2,rd); r=min(ks)
            if r in seen: continue
            seen.add(r); res.append((ms1,ms2,k))
print("k dist (pairs, all):",dict(sorted(kdist.items())),flush=True)
print("classes with k in",kt,":",len(res),round(time.time()-t),"s",flush=True)
pickle.dump(res,open(out,'wb'))
