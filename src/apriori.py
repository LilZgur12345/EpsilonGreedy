import numpy as np
import pandas as pd
from apyori import apriori

def run_apriori():
  order_data = pd.read_csv('data/order_products_train.csv')
  product_data = pd.read_csv('data/products.csv')

  named_orders = pd.merge(order_data,product_data,on='product_id')
  named_orders.sort_values('product_id', inplace=True)

  counts = named_orders['product_name'].value_counts()

  t = np.linspace(25,95,num=1000)
  cost = []
  data = np.log(counts)
  for i in t:
    data1 = data[data<np.percentile(data,i)]
    data2 = data[data>=np.percentile(data,i)]
    cost.append(i*np.var(data1)+(100-i)*np.var(data2))
  v = counts[np.argmin(cost)]

  p = t[np.argmin(cost)]
  v=200
  counts = counts[counts > v]
  selected_orders = named_orders[named_orders['product_name'].isin(counts.index.values.tolist())]

  pd.options.mode.chained_assignment = None
  selected_orders['cols'] = selected_orders.groupby('order_id').cumcount()
  selected_pivot = selected_orders.pivot(index = 'order_id',columns = 'cols')[['product_name']]

  purchases = []
  for i in range(0,len(selected_pivot)):
    purchases.append([str(selected_pivot.values[i,j]) for j in range(0,41)])

  cleanedList = []
  for i in range(len(purchases)):
    cleanedList.append([x for x in purchases[i] if str(x) != 'nan'])

  rules = apriori(purchases, min_support = 0.005, min_confidence = 0.005, min_lift=3,max_length=2)
  results = list(rules)

  for item in results:
      pair = item[0]
      if "nan" not in pair:
          if item[2][0][2] > item[2][1][2]:
            items = [x for x in pair]
            print("Rule: " + items[0] + " => " + items[1])
            print("Support: " + str(round(item[1]*100,2)) + "%")
            print("Confidence: " + str(round( item[2][0][2]*100,2)) + "%")
            print("Lift: " + str( round(item[2][0][3],2) ))
          else:
            items = [x for x in pair]
            print("Rule: " + items[1] + " => " + items[0])
            print("Support: " + str(round(item[1]*100,2)) + "%")
            print("Confidence: " + str(round( item[2][1][2]*100,2)) + "%")
            print("Lift: " + str( round(item[2][1][3],2) ))

  length = len(results)
  print("I found " + str(length) + " rules")