import sys,time,json,ast; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
# usage: k6_narrow.py TAG L DMIN  (reads k6_TAG.csv; narrow symmetry class)
import pandas as pd
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_fullgroup import narrow_group_mats, nilradical_certificate
from bb_logical_group import classify_logical_group, clusters_of
tag=sys.argv[1]; L=eval(sys.argv[2]); dmin=int(sys.argv[3]); out=f"k6_{tag}_narrow.jsonl"
rd=ring_setup(L)
df=pd.read_csv(f'k6_{tag}.csv'); df=df[(df.status=='evaluated')&(df.d>=dmin)]
done=set()
try: done={json.loads(l)['module_key'] for l in open(out)}
except FileNotFoundError: pass
for _,r in df.iterrows():
    if r.module_key in done: continue
    t=time.time()
    f1=RingElement.from_string(L,r.f1); f2=RingElement.from_string(L,r.f2)
    mats,k,info=narrow_group_mats(f1,f2,rd)
    cert=nilradical_certificate(mats,info['trans'],L)
    rep=classify_logical_group(mats,k,max_points=4200)
    cl=[(c['m'],c['order'],c.get('full')) for c in rep['clusters']]
    row=dict(module_key=r.module_key,L=str(L),f1=r.f1,f2=r.f2,n=int(r.n),k=int(r.k),d=int(r.d),semilinear=info["semilinear"],fold=info["fold"],generators=len(mats),order=str(rep['order']),status=rep['order_status'],clusters=str(cl),full=bool(rep['acts_as_full_clifford']),nilcert=str(cert),seconds=round(time.time()-t))
    open(out,'a').write(json.dumps(row)+'\n'); print(row,flush=True)
