import sys,json,pickle; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_fullgroup import narrow_group_mats
L=(5,5); rd=ring_setup(L)
def nullity(rows,n):
    M=[int(''.join(map(str,r)),2) for r in rows]; rank=0
    for bit in range(n-1,-1,-1):
        p=next((i for i in range(rank,len(M)) if M[i]>>bit&1),None)
        if p is None: continue
        M[rank],M[p]=M[p],M[rank]
        for i in range(len(M)):
            if i!=rank and M[i]>>bit&1: M[i]^=M[rank]
        rank+=1
    return n-rank
for l in list(open('ff_55_k8s.jsonl'))[:3]:
    r=json.loads(l); f1=RingElement.from_string(L,r['f1']); f2=RingElement.from_string(L,r['f2'])
    mats,k,info=narrow_group_mats(f1,f2,rd); n=2*k
    mats=[np.array(m,dtype=np.uint8)%2 for m in mats]
    # X with gX = X g  <=> (g^T kron I + I kron ... ) vec; build rows of linear system over n*n unknowns
    rows=[]
    for g in mats:
        for i in range(n):
            for j in range(n):
                row=np.zeros(n*n,dtype=np.uint8)
                # (gX)_{ij}=sum_t g[i,t]X[t,j]; (Xg)_{ij}=sum_t X[i,t]g[t,j]
                for t in range(n):
                    if g[i,t]: row[t*n+j]^=1
                    if g[t,j]: row[i*n+t]^=1
                rows.append(row)
    print(r['f1'],'| commutant dim',nullity([list(x) for x in rows],n*n),'info',info,flush=True)
