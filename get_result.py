import pandas as pd
import pickle


df = pd.read_csv('1180__2019.csv', sep=';', index_col=[0])
df.fillna(0, inplace=True)


with open('data.pickle', 'rb') as file:
	net = pickle.load(file)


answ = []

for i in df[['hdr', 'steps', 'energy_in', 'energy_out', 'heart_rate', 'stress_level']].values.tolist():
	answ.append(net.activate(i)[0])

df['answ'] = answ



a = 0

for i in df[['anxiety_level', 'answ']].values.tolist():
	print(i)
	a += abs(round(i[0]+0.5)-round(i[1]))

print('Точность сотовляет', 100-a/528648)

df.to_csv('123.csv', sep=';', index=False)