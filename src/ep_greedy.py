import numpy as np
from scipy.stats import uniform
from src.machines import casino

def run_epsilon_greedy():
  cost = 2
  money = 200
  max_number_of_plays = 200
  c = 2
  epsilon = 0.2
  d = 15
  n = [0] * d
  R = [0] * d
  k = 0
  machine_record = []

  for i in range(d):
    reward = casino(i+1)
    R[i] = reward
    n[i] += 1
    k = i + 1
    money = money + reward - cost
    machine_record.append(i+1)

  while (k < max_number_of_plays) & (money > 0):
    random_num = uniform.rvs()
    if random_num < epsilon:
      star = np.random.randint(d)
    else:
      star = np.argmax([R[i] / n[i] for i in range(d)])
    n[star] += 1
    k += 1
    R[star] = R[star] + casino(star+1)
    money = money + R[star] - cost
    machine_record.append(star+1)

# UCB is more effective/accurate for solving two-handed bandit problems like this
  lucky = np.argmax(n)+1
  print("The lucky machine is " + str(lucky))