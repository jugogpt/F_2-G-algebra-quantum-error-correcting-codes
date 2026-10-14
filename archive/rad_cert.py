import sys,pickle; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
import numpy as np
def rank_cols(cols):
    basis={};r=0
    for v in cols:
        while v:
            h=v.bit_length()-1
            if h in basis: v^=basis[h]
            else: basis[h]=v;r+=1;break
    return r,basis
def colints(M): 
    M=np.asarray(M)%2; return [int("".join(str(int(M[i,j])) for i in range(M.shape[0]))[::-1],2) for j in range(M.shape[1])]
def apply_int(M,v):
    M=np.asarray(M)%2; n=M.shape[0]; out=0
    for j in range(n):
        if v>>j&1: out^=int("".join(str(int(M[i,j])) for i in range(n))[::-1],2)
    return out
def radical_cert(mats,trans):
    n=trans[0].shape[0]; I=np.eye(n,dtype=np.uint8)
    gens=[]
    for T in trans: gens+=colints((np.asarray(T)^I)%2)
    r,basis=rank_cols(gens); W=list(basis.values())
    inv=True
    for M in mats:
        imgs=[apply_int(M,w) for w in W]
        r2,_=rank_cols(W+imgs)
        if r2!=r: inv=False;break
    return {"dim_V":n,"dim_radV":r,"invariant":inv}
if __name__=="__main__":
    m=[np.array(x) for x in pickle.load(open('/home/claude/runs/gross_all_mats.pkl','rb'))]
    tr=pickle.load(open('/home/claude/runs/gross_translations.pkl','rb'))
    T=[np.array(v) for v in tr.values()]
    print("gross:",radical_cert(m+T,T))
