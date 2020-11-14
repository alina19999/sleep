import os
import neat
import pickle
import pandas as pd
import time



df = pd.read_csv('1180__2019.csv', sep=';')
df.fillna(0, inplace=True) 


inputs = df[['hdr', 'steps', 'energy_in', 'energy_out', 'heart_rate', 'stress_level']].values.tolist()
outputs = df[['anxiety_level']].values.tolist()


count = 0

def eval_genomes(genomes, config):  #функция тренировки каждого поколения
    global count, start
    for genome_id, genome in genomes:
        count += 1
        print(count) #вывод какая по счету нейросеть тренируется
        genome.fitness = 4.0 #изначально значение fitness для каждой нейросети равно 4, чем больше значение fitness, тем она более пригодна
        net = neat.nn.FeedForwardNetwork.create(genome, config) #создается нейросеть
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi) #получение ответа от нейросети для каждого варианта входных значений
            genome.fitness -= abs(round(xo[0]+0.5)-output[0]) #от fitness отнимается разница между предпологаемым ответом и ее ответом


def run(config_file):#загружается конфигурация neat
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    #создается популяция для последующей тренировки
    p = neat.Population(config)

    #лучший вариант записывается в переменную winner, указывается функция тренировки и количество поколений
    winner = p.run(eval_genomes, 50)


    print('\nOutput:')
	#нейросеть с самым высоким показателем точности записывается в winner_net
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
	#выводятся все ответы нейросети для быстрой проверки
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("input {!r}, output {!r}, got {!r}".format(xi, xo, output[0]))

		
	#лучшая нейросеть записывается в файл data.pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump(winner_net, f)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__) #получение текущей папки
    config_path = os.path.join(local_dir, 'config-feedforward') #записывается путь к конфигурационному файлу
    run(config_path) #запуск