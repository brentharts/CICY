import itertools, numpy as np, sympy as sp
from sympy import Matrix, eye, symbols
from functools import reduce
from math import gcd
from pyCICY import sixsphere_search as Q
from pyCICY import sixsphere as S
fn=Q.funnel(5); cls=Q.classify(fn['survivors'])
M=lambda a: Matrix(a.tolist()) if hasattr(a,'tolist') else Matrix(a)
reps=[('paper',(S.T1,S.T2))]+[(str(sorted(c['invariant'])),(M(c['rep'][0]),M(c['rep'][1]))) for c in cls]
def primQ(T1,T2):
    q=symbols('q0:6'); Qm=S._antisym(q); eqs=[]
    for T in (T1,T2): eqs+=list(T.T*Qm*T-Qm)
    sol=list(sp.linsolve(eqs,q))[0]; s=list(set().union(*[sp.sympify(e).free_symbols for e in sol]))[0]
    v=[sp.nsimplify(e.subs(s,1)) for e in sol]; den=sp.ilcm(*[sp.Rational(e).q for e in v]); v=[int(e*den) for e in v]
    g=reduce(gcd,[abs(e) for e in v]); return S._antisym([e//g for e in v])
def adapted(Qm):
    Qn=np.array(Qm.tolist(),dtype=np.int64); Q0=np.array(S.Q0.tolist(),dtype=np.int64)
    V=np.array(list(itertools.product(range(-3,4),repeat=4)),dtype=np.int64)
    V=V[np.any(V!=0,axis=1)]
    QV=V@Qn
    for i,e0 in enumerate(V):
        ok=np.where(QV[i]@V.T==1)[0]   # Q(e0,e3)=1  (row e0^T Q v)
        for j in ok:
            e3=V[j]
            C=V[(QV[i]@V.T==0)&((V@Qn@e3)==0)]  # orthogonal to e0 and e3 (Q(e0,v)=0=Q(e3,v))
            if len(C)<2: continue
            Ci=(C@Qn)
            for a in range(len(C)):
                w=np.where(Ci[a]@C.T==6)[0]
                for b in w:
                    P=np.stack([e0,C[a],C[b],e3],axis=1)
                    if abs(round(np.linalg.det(P)))==1:
                        Pm=Matrix(P.tolist())
                        if Pm.T*Qm*Pm==S.Q0: return Pm
    return None

for n,(a,b) in reps:
    P=eye(4) if n=='paper' else adapted(primQ(a,b))
    if P is None: print(n,'no adapted basis (box 3)',flush=True); continue
    t1,t2=P.inv()*a*P,P.inv()*b*P; t0=(t1*t2).inv()
    L={g:tuple(sp.cancel(x) for x in S.laws_from_lattice(T)['law']) for g,T in (('g1',t1),('g2',t2),('g0',t0))}
    X=(S.TAU,S.MU,S.BETA)
    triv1=S.law(['g1']*3,L)==X; triv2=S.law(['g2']*4,L)==X
    g12=S.law(['g1','g2'],L); inv0=S.law(['g0'],L)
    # g0 = (g1 g2)^-1 : law(g0 o g1g2) must be identity
    comp=S.law(['g0','g1','g2'],L)==X
    print(n,'g1^3 trivial',triv1,'g2^4 trivial',triv2,'g0 g1 g2 = id',comp,flush=True)
