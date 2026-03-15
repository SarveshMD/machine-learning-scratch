from sympy import symbols, diff

X = 2
Y = 10
alpha = 0.01
mval = 1
cval = 0

m,c,x = symbols('m c x')

y = m*x + c
E = (y-Y)**2

dE_dm = diff(E, m)
dE_dc = diff(E, c)

for i in range(400):
    # print(f"\n--- ITERATION {i+1} ---")
    dE_dm_val = dE_dm.subs({x:2,m:mval,c:cval})
    dE_dc_val = dE_dc.subs({x:2,m:mval,c:cval})

    # print(dE_dm_val)
    # print(dE_dc_val)

    mval = mval - (alpha * dE_dm_val)
    cval = cval - (alpha * dE_dc_val)

    # print(mval, cval)
    # print(f"ycap: {y.subs({x:2,m:mval,c:cval})}")
    # print(f"E: {E.subs({x:2,m:mval,c:cval})}")

print(mval, cval)
print(f"ycap: {y.subs({x:2,m:mval,c:cval})}")
print(f"E: {E.subs({x:2,m:mval,c:cval})}")