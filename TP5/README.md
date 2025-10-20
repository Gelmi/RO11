# ROB311 - TP Reinforcement Learning

## Definitions

- **States:** $S=\\{S_0,S_1,S_2,S_3\\}$
- **Valid actions:**
  $$A(S_0)=\\{a_1,a_2\\}, A(S_1)=A(S_2)=A(S_3)=\\{a_0\\}$$
- **Rewards:** $$R(S_3)=10, R(S_2)=1, R(S_0)=R(S_1)=0$$
- **Discount factor:** $$\gamma\in(0,1)$$
- **Parameters:** $$x,y\in[0,1]$$

### Transition matrices $$T(s,a,s')$$
(rows = current state; columns = next state $S'$)

- For $a_0$:

$$
  T(S,a_0,S')=
  \begin{bmatrix}
  0&0&0&0\\
  0&1-x&0&x\\
  1-y&0&0&y\\
  1&0&0&0
  \end{bmatrix}
$$

- For $a_1$:

$$
  T(S,a_1,S')=
  \begin{bmatrix}
  0&1&0&0\\
  0&0&0&0\\
  0&0&0&0\\
  0&0&0&0
  \end{bmatrix}
$$

- For $a_2$:

$$
  T(\cdot,a_2,\cdot)=
  \begin{bmatrix}
  0&0&1&0\\
  0&0&0&0\\
  0&0&0&0\\
  0&0&0&0
  \end{bmatrix}
$$

> **Optimal Bellman:**  
> $` V^*(s)=R(s)+\gamma\max_{a\in A(s)}\sum_{s'}T(s,a,s')\,V^*(s') `$

---

## Q1 — Enumerate all possible policies

Since only $S_0$ has two actions, there are **two** stationary deterministic policies:

1. $`\pi_1:\ S_0\!\to\! a_1;\ S_1,S_2,S_3\!\to\! a_0`$
2. $`\pi_2:\ S_0\!\to\! a_2;\ S_1,S_2,S_3\!\to\! a_0`$

---

## Q2 — Equations for $$\(V^*(s)\)$$

Using $R(s)$ and $T$:

$` V^*(S_0) = \gamma \max \{ V^*(S_1),\ V^*(S_2)\} `$

$`V^*(S_1) = \gamma[(1-x)\,V^*(S_1)+x\,V^*(S_3)]`$

$`V^*(S_2) = 1+\gamma[(1-y)\,V^*(S_0)+y\,V^*(S_3)]`$

$`V^*(S_3) = 10+\gamma\,V^*(S_0)`$



---

## Q3 — Does there exist $x$ such that, $\forall\,\gamma\in(0,1)$ and $\forall\,y\in[0,1]$, $\pi^*(S_0)=a_2$?

**Yes.** Choose **$x=0$**.  
Then $S_1$ becomes a sink with no reward:

$$
V^* (S_1) = \gamma [(1-0)V^\*(S_1) + 0 \cdot V^\*(S_3)] \Rightarrow V^\*(S_1)=0
$$

Since $V^\*(S_2)\ge 1$ (immediate reward in $S_2$), we have $V^\*(S_2)\ge V^\*(S_1)$ for any $\gamma,y$. Therefore, $\pi^*(S_0)=a_2$.

---

## Q4 — Does there exist $y$ such that, $\forall\,x>0$ and $\forall\,\gamma\in(0,1)$, $\pi^*(S_0)=a_1$?

**No.** As $\gamma\to 0$, the optimal criterion prioritizes only immediate reward:  
$V^\*(S_1)\to 0$ and $V^\*(S_2)\to 1$.  
Therefore, for sufficiently small $\gamma$, $a_2$ dominates at $S_0$, regardless of $y$ and any $x>0$. Thus, such a $y$ does not exist.

---

## Q5 — Optimal value and policy for $x=y=0{,}25$, $\gamma=0{,}9$

With value iteration (stopping rule $\max_s|V_k(s)-V_{k-1}(s)|<10^{-4}\$:

- $V^\*(S_0)\approx 14{,}1856$
- $V^\*(S_1)\approx 15{,}7618$
- $V^\*(S_2)\approx 15{,}6979$
- $V^\*(S_3)\approx 22{,}7671$

Optimal action at $S_0$:

$$
Q(S_0,a_1)=\gamma V^\*(S_1)\approx 14{,}1856,\quad
Q(S_0,a_2)=\gamma V^\*(S_2)\approx 14{,}1281\ \Rightarrow\ \boxed{\pi^\*(S_0)=a_1}.
$$

In the other states, the only available action is $a_0$.
