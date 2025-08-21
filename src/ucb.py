from src.machines import casino
import numpy as np
import matplotlib.pyplot as plt

# Upper Confidence Bound algorithm (multi-armed bandit)

def run_ucb():
  cost = 2
  money = 200
  max_number_of_plays = 200
  d = 15
  n = [0] * d
  R = [0] * d
  k = 0
  machine_record = []
  UCB = [0] * d
  c = 2 # assume this is true

  for i in range(d):
    reward = casino(i+1)
    R[i] = reward
    n[i] += 1
    k += 1
    money = money + reward - cost
    machine_record.append(i+1)
    UCB[i] = R[i] +c*np.sqrt(np.log(k)/n[i])

  while (k < max_number_of_plays) & (money>cost):
    star = np.argmax(UCB)
    reward = casino(star+1)
    R[star] = R[star] + reward
    money = money + reward - cost
    n[star] += 1
    k += 1

    UCB[star] = R[star]/n[star] +c*np.sqrt(np.log(k)/n[star])
    machine_record.append(star+1)

  lucky = np.argmax(n)+1
  print("The lucky machine is " + str(lucky))

  plt.hist(machine_record,bins=range(22),rwidth=0.9, alpha=0.5,align='left')
  plt.title("Machine Selection (End Balance: "+ str(round(money,2))+")")
  plt.xlabel("Slot Machine")
  plt.ylabel("Number of Times Used")
  plt.xticks(np.arange(1,22,1))
  plt.show()