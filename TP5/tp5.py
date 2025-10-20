import numpy as np
gamma = 0.9
x = y = 0.25
V = np.zeros(4)

while True:
    Vprev = V.copy()
    V3 = 10 + gamma*Vprev[0]
    V1 = gamma*((1-x)*Vprev[1] + x*V3)
    V2 = 1 + gamma*((1-y)*Vprev[0] + y*V3)
    V0 = gamma*max(V1, V2)
    V = np.array([V0, V1, V2, V3])
    if np.max(np.abs(V - Vprev)) < 1e-4:
        break

policy_S0 = 'a1' if V[1] >= V[2] else 'a2'
print(V, policy_S0)