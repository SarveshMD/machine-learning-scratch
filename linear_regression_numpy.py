import numpy as np

X = np.array([1,3,5])
Y = np.array([2,4,3])
alpha = 0.01
mval = 1
cval = 0

for i in range(10000):
    # print(f"\n--- ITERATION {i+1} ---")

    y = mval*X + cval
    E = (y-Y)**2

    dE_dm = np.mean(2*X*(y-Y))
    dE_dc = np.mean(2*(y-Y))

    mval = mval - (alpha * dE_dm)
    cval = cval - (alpha * dE_dc)

    # print(f"dE/dm: {dE_dm}")
    # print(f"dE/dc: {dE_dc}")
    # print(f"Emean: {np.mean(E)}")
    # print(f"E: {E}")
    # print(f"ycap: {y}")
    # print(f"m: {mval}\nc: {cval}")

print("\nFINAL:")
print(f"dE/dm: {dE_dm}")
print(f"dE/dc: {dE_dc}")
print(f"Emean: {np.mean(E)}")
print(f"ycap: {y}")
print(f"m: {mval}\nc: {cval}")

# PLOTTING THE RESULT

import matplotlib.pyplot as plt

xpts = np.linspace(0,10,100)
ypts = mval*xpts + cval

fig, ax = plt.subplots()

plt.xlim(0,10)
plt.ylim(0,10)
plt.axis('equal')
plt.grid(True)

ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,11,1))

ax.plot(xpts, ypts, c='black')
ax.scatter(X,Y, c='black')

plt.show()