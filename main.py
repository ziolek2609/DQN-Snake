from numpy import argmax
from random import random, randint
import matplotlib.pyplot as plt
from dqn import Dqn
from neural_network import NeuralNetwork
from snake_environment import SnakeEnvironment


# STAŁE PARAMETRY UCZENIA
# liczba epok treningowych
epochs = 5000
# parametr dyskontujący do dqn
gamma = 0.9
# zmiana epsilon po każdej grze
epsilonMultiplier = 0.999
# minimalna wartość prawdopodobnieństw ruchu losowego
minEpsilon = 0.05
# pojemnosc pamieci
maxMemory = 10000
# learning rate
learningRate = 0.005
# wielkosć batcu do sieci
batchSize = 256
# nazwa modelu
model_name = "model.h5"


# USTAWIENIA ŚRODOWISKA
# czas pomiędzy przesunięciami węża
waitTime = 0
# ilość segmentów na jednym boku ekranu
nSegments = 4
# czy pokazywać wizualizacje w pygame?
visualization = False
# zestaw nagród
rewards = [-0.02, 1, -4]
# koordynaty: 1 -- bezwzględne, 2 -- względne
cordinate = 1
# wizja: 1 -- odległosci, 2 -- obecnosci
vision = 1


# STWORZENIE ŚRODOWISKA, MODELU SIECI ORAZ DQN
env = SnakeEnvironment(waitTime=waitTime, segments=nSegments,
                       vision=vision, livingPenalty=rewards[0],
                       posReward=rewards[1], negReward=rewards[2],
                       cordinate=cordinate, visualization=visualization)
if cordinate == 1:
    nn = NeuralNetwork(24, 4, learningRate)
elif cordinate == 2:
    nn = NeuralNetwork(21, 3, learningRate)
model = nn.model
DQN = Dqn(gamma, maxMemory)


# ZMIENNE WYKORZYSTYWANE W PROCESIE UCZENIA
epsilon = 1
epoch = 1
scoreInEpochs = []
meanScore = 0
bestMeanScore = 0
bestScore = [0, 0]
fullMemoryEpoch = 0
win = False
winNum = 0

while epoch <= epochs:
    # NOWA GRA -- reset środowiska, i początkowy input
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False

    while not gameOver:
        # ustalenie czy podejmowana akcja będzie losowa czy predykowana
        if random() <= epsilon:
            if cordinate == 1:
                action = randint(0, 3)
            elif cordinate == 2:
                action = randint(0, 2)
        else:
            action = argmax(model.predict(currentState))

        # podjęcie akcji
        nextState, reward, gameOver, win = env.step(action)

        # zliczanie zwycięstw
        if win:
            winNum += 1

        # umieszczenie ruchu w pamięci i trening sieci na pobranym batchu
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState

    # zmniejszenie prawdopodobieństwa losowości
    if epsilon > minEpsilon:
        epsilon *= epsilonMultiplier
    else:
        epsilon = minEpsilon

    # statystyki z pojedynczej gry
    if env.score > bestScore[0] or (env.score == bestScore[0] and
                                    env.moves < bestScore[1]):
        bestScore = [env.score, env.moves]
        print("NEW BEST SCORE:", bestScore[0], "points in",
              bestScore[1], "moves")
    print("Epoch:\t", epoch, "Score:\t", env.score, "Moves:\t", env.moves,
          "Epsilon:\t", round(epsilon, 5), "Best score:", bestScore[0])
    meanScore += env.score

    # co 100 epok -- statystyka ze 100 epok
    if epoch % 100 == 0:
        scoreInEpochs.append(meanScore/100)
        print("MEAN SCORE IN LAST 100 EPOCHS:", meanScore/100,
              "ALL TIME BEST SCORE:", bestScore[0], "in", bestScore[1],
              "moves. ACTUAL MEMORY CAPACITY:", len(DQN.memory))
        plt.plot(scoreInEpochs)
        plt.xlabel('Epoki*100')
        plt.ylabel('Średni wynik w 100 epokach')
        plt_name = "model_unrelative_distance.jpg"
        plt.savefig(plt_name)
        plt.show()
        plt.clf()
        plt.cla()

        if meanScore/100 > bestMeanScore:
            bestMeanScore = meanScore/100
        meanScore = 0

    # zapisanie epoki, w której pamięć się zapełniła
    if len(DQN.memory) < maxMemory:
        fullMemoryEpoch = epoch+1

    epoch += 1

# zapisanie modelu
model.save(model_name)
