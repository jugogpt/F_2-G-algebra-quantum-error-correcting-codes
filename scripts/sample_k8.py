import sys,random,pickle,collections; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
# usage: sample_k8.py SEED MAX_WEIGHT N_SAMPLES OUT.pkl
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_sweep import fast_dimension
from bb_equiv import equivalence_keys
L=(5,5); rd=ring_setup(L)
# GF(16), x^4=x+1
def mul(a,b):
    r=0
    for i in range(4):
        if b>>i&1: r^=a<<i
    for i in (7,6,5,4):
        if r>>i&1: r^=0b10011<<(i-4)
    return r
def pw(a,n):
    r=1
    for _ in range(n): r=mul(r,a)
    return r
z=pw(2,3); assert pw(z,5)==1 and z!=1
# D=4 components: orbits of (a,b) in Z5^2\0 under *2 ; reps
seen=set(); reps=[]
for a in range(5):
    for b in range(5):
        if (a,b)==(0,0) or (a,b) in seen: continue
        orb={((a*2**i)%5,(b*2**i)%5) for i in range(4)}; seen|=orb; reps.append((a,b))
pts=[(0,0)]+reps
def ev(ms,p):
    s=0
    for (i,j) in ms: s^=pw(z,(p[0]*i+p[1]*j)%5)
    return s
mons=[(i,j) for i in range(5) for j in range(5)]
random.seed(int(sys.argv[1])); W=int(sys.argv[2]); N=int(sys.argv[3])
res=[];seen_k=collections.Counter();keys=set()
for t in range(N):
    c=random.choice(reps)
    def samp():
        while True:
            ms=[(0,0)]+random.sample(mons[1:],random.randint(2,W-1))
            if ev(ms,c)==0: return ms
    a,b=samp(),samp()
    if any(ev(a,p)==0 and ev(b,p)==0 for p in pts if p!=c): continue
    k=fast_dimension(RingElement(L,a),RingElement(L,b),rd); seen_k[k]+=1
    if k==8:
        key=min(equivalence_keys(RingElement(L,a),RingElement(L,b),rd))
        if key in keys: continue
        keys.add(key); res.append((a,b,8))
print(seen_k,len(res)); pickle.dump(res,open(sys.argv[4],'wb'))
