import sympy as sp
from sympy import Matrix, eye
from sympy.matrices.normalforms import invariant_factors
from pyCICY import sixsphere_search as Q, sixsphere as S
fn=Q.funnel(5); cls=Q.classify(fn['survivors'])
M=lambda a: Matrix(a.tolist()) if hasattr(a,'tolist') else Matrix(a)
reps=[('paper',(S.T1,S.T2))]+[(str(sorted(c['invariant'])),(M(c['rep'][0]),M(c['rep'][1]))) for c in cls]
for n,(a,b) in reps:
    T0=(a*b).inv(); N=T0-eye(4)
    f=[abs(int(x)) for x in invariant_factors(N)]
    K=S.int_kernel(N)          # saturated kernel basis (rows)
    Im=S.saturate_basis(N.T)   # saturation of image
    # index of image lattice in ker N: smith of image coords wrt ker basis
    cols=N  # image spanned by columns of N
    Kb=Matrix(K).T if K.shape[0]==2 else Matrix(K)
    # express columns of N in kernel basis
    sol=Kb.pinv()*N if False else (Kb.T*Kb).inv()*Kb.T*N
    idx=abs(Matrix(sol).applyfunc(sp.nsimplify)[:, :].rank()) 
    ff=[abs(int(x)) for x in invariant_factors(Matrix(sol))]
    print(n,'smith(N)',f,'im N in ker N: factors',ff,'ker==satim',S.same_lattice(K,Im))
