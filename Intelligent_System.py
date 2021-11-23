import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv',
                   delimiter = '\t')

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None


# 1
print('Всего наблюдений', data.shape[0])

# 2
print('Столбцы:', end = ' ')
print(data.columns.tolist())
print()

# 3
print('Самая частая позиция:',
      data['item_name'].value_counts().idxmax())

# 4
data['item_name'].value_counts().plot.bar()
plt.show()

# 5
data.item_price = data['item_price'].apply(lambda x: float(x[1:]))

# 6
data[['item_name', 'item_price']].groupby('item_name').sum()['item_price'].plot.bar()
plt.show()

# 7
print('Средняя сумма заказа:')
print('способ 1:',
      (data[['order_id', 'item_price']].groupby('order_id').sum().sum() /
       data.order_id.max()).tolist()[0])
print('способ 2:',
      data[['order_id', 'item_price']].groupby('order_id').sum().mean().tolist()[0])

# 8
print('Количество позиций в заказе:')
print(data[['order_id']].value_counts().agg(['max', 'min', 'mean', 'median']))

# 9
steak = data.loc[data['item_name'].str.contains('Steak')]
print('Статистика заказов стейков:',
      steak.groupby('item_name')[['item_price', 'quantity']].describe())
steak_mild = steak.loc[steak['choice_description'].str.contains('Mild')]
steak_hot = steak.loc[steak['choice_description'].str.contains('Hot')]
steak_medium = steak.loc[steak['choice_description'].str.contains('Medium')]
print('Статистика заказов прожарки Mild:',
      steak_mild.groupby('item_name')[['item_price', 'quantity']].describe())
print('Статистика заказов прожарки Medium:',
      steak_medium.groupby('item_name')[['item_price', 'quantity']].describe())
print('Статистика заказов прожарки Hot:',
      steak_hot.groupby('item_name')[['item_price', 'quantity']].describe())

# 10
new_column = data[['item_price']].apply(lambda x: x*71.41).rename(columns = {'item_price': 'item_price_rub'})
data = data.merge(new_column, left_index = True, right_index = True)

#11
print(data['item_name'].value_counts())
print('Отдельная группировка по стейкам:')
print (steak['item_name'].value_counts())

# 12
prices = []
for id, row in data.iterrows():
    if 'and' not in row['item_name']:
        prices.append(row['item_name'] + ' cost '
                      + str(round(row['item_price'] / row['quantity'], 2)) + '$ or '
                      + str(round(row['item_price_rub'] / row['quantity'], 2)) + 'P')            

prices = list(set((prices)))
for elem in prices:
    print(elem)
