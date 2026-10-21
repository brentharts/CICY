import itertools, sympy as sp
from sympy import Matrix, symbols, linsolve, eye, zeros
from pyCICY import sixsphere_search as Q
fn=Q.funnel(5); cls=Q.classify(fn['survivors'])
def anti(q): return Matrix([[0,q[0],q[1],q[2]],[-q[0],0,q[3],q[4]],[-q[1],-q[3],0,q[5]],[-q[2],-q[4],-q[5],0]])
for c in cls:
    T1,T2=[Matrix(x.tolist()) if hasattr(x,'tolist') else Matrix(x) for x in c['rep']]
    q=symbols('q0:6'); Qm=anti(q)
    eqs=[]
    for T in (T1,T2): eqs+=list(T.T*Qm*T-Qm)
    sol=list(linsolve(eqs,q))[0]
    free=sorted(set().union(*[sp.sympify(x).free_symbols for x in sol]),key=str)
    T0=(T1*T2).inv(); N=T0-eye(4)
    print('class',sorted(c['invariant']),'N^2=0',N*N==zeros(4),'rankN',N.rank(),'free params',free)
    Qg=anti(sol)
    print('  Q general:',Qg.tolist())
    # b = Q N ; find symmetric/definite conditions
    B=Qg*N
    print('  QN symmetric cond:',[sp.simplify(e) for e in (B-B.T) if sp.simplify(e)!=0][:4])
    if len(free)==1:
        s=free[0]; Bn=B.subs(s,1)
        K=N.nullspace(); 
        # restrict to complement: use quadratic form x^T B x on all V, signature of sym part
        S=(Bn+Bn.T)/2
        print('  sym part eigen:',S.eigenvals(), ' det Q (pf^2):', Qg.subs(s,1).det())
print('--- primitive Q, Pfaffian, b on V/kerN')
from math import gcd
from functools import reduce
for c in cls:
    T1,T2=[Matrix(x.tolist()) if hasattr(x,'tolist') else Matrix(x) for x in c['rep']]
    q=symbols('q0:6'); Qm=anti(q); eqs=[]
    for T in (T1,T2): eqs+=list(T.T*Qm*T-Qm)
    sol=list(linsolve(eqs,q))[0]; s=symbols('q5')
    v=[sp.nsimplify(x.subs(s,1)) for x in sol]
    den=sp.ilcm(*[sp.Rational(x).q for x in v]); v=[int(x*den) for x in v]
    g=reduce(gcd,[abs(x) for x in v]); v=[x//g for x in v]
    Qp=anti(v); pf=v[0]*v[5]-v[1]*v[4]+v[2]*v[3]
    N=((T1*T2).inv()-eye(4)); B=Qp*N
    # b on V/kerN: pick complement basis via Smith-free approach: columns of N span im; use rows quotient
    K=N.nullspace(); Kz=[ (k*sp.ilcm(*[x.q for x in k])) for k in K]
    # saturated kernel basis -> extend to unimodular basis
    M=Matrix.hstack(*Kz); 
    import itertools
    best=None
    for cand in itertools.product(range(-1,2),repeat=8):
        W=Matrix.hstack(M,Matrix(4,2,cand))
        if abs(W.det())==1: best=W;break
    G=(best.T*B*best)[2:,2:] if best is not None else None
    print(sorted(c['invariant']),'Q prim',v,'Pf',pf,'b gram',None if G is None else G.tolist(),'disc',None if G is None else G.det(), 'N Q-sign:', )
