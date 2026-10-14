# usage: find_full.py "(3,3)" "[4]" "[(3,3),(3,4),(4,3),(4,4),(2,4),(4,2),(2,3),(3,2),(2,2)]" out.jsonl
import sys,time,json,pickle,subprocess; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','src'))
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_fullgroup import narrow_group_mats, full_group_mats, nilradical_certificate
from bb_logical_group import classify_logical_group, clusters_of
from bb_prove_full import prove_full
from bb_distance import code_distance
from bb_equiv import equivalence_keys
L=eval(sys.argv[1]); K=sys.argv[2]; W=sys.argv[3]; out=sys.argv[4]; wide=len(sys.argv)>5 and sys.argv[5]=='wide'
tag=out.replace('.jsonl','')
pass
cands=pickle.load(open(tag+'.pkl','rb')); rd=ring_setup(L)
ex5=None
if L==(3,3):
    ex5=equivalence_keys(RingElement.from_string(L,"1 + x + y"),RingElement.from_string(L,"1 + y + x^2*y"),rd)
done=set()
try: done={json.loads(l)['f1']+'|'+json.loads(l)['f2'] for l in open(out)}
except FileNotFoundError: pass
for a,b,k in cands:
    f1,f2=RingElement(L,a),RingElement(L,b)
    if str(f1)+'|'+str(f2) in done: continue
    t=time.time(); row=dict(L=str(L),f1=str(f1),f2=str(f2),k=k,n=2*rd.DL())
    try:
        d=code_distance(f1,f2,rd,time_limit=600); row.update(d=d['d'],exact=d['exact'])
        mats,kk,info=(full_group_mats if wide else narrow_group_mats)(f1,f2,rd)
        rep=classify_logical_group(mats,kk,max_points=4200 if kk<=6 else 100)
        row.update(generators=len(mats),order=str(rep['order']),clusters=str([(c['m'],c['order']) for c in rep['clusters']]),exact_full=bool(rep['acts_as_full_clifford']))
        pf=prove_full(mats,kk) if len(rep['clusters'])==1 and rep['clusters'][0]['m']==kk else {'verdict':'multi_cluster'}
        row.update(proof=pf['verdict'],full=(row['exact_full'] or pf['verdict']=='full'))
        row['same_as_example5']=bool(ex5 and (ex5 & equivalence_keys(f1,f2,rd)))
    except Exception as e:
        row['error']=f"{type(e).__name__}:{e}"[:150]
    row['seconds']=round(time.time()-t)
    open(out,'a').write(json.dumps(row)+'\n'); print(row,flush=True)
