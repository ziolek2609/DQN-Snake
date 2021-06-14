from numpy import argmax
import matplotlib.pyplot as plt
from dqn import Dqn
from snake_environment import SnakeEnvironment
from keras.models import load_model


# STAŁE PARAMETRY UCZENIA
# liczba epok dotrenowania
epochs = 10000
# parametr dyskontujący do dqn
gamma = 0.9
# pojemnosc pamieci
maxMemory = 50000
# learning rate
learningRate = 0.005
# wielkosc batchu do sieci
batchSize = 256
# nazwa modelu do dotrenowania
pretrained_model = "model_copy.h5"


# USTAWIENIA ŚRODOWISKA
# czas pomiędzy przesunięciami węża
waitTime = 0
# ilość segmentów na jednym boku ekranu
nSegments = 8
# czy pokazywać wizualizację w pygame?
visualization = False
# zestaw nagród
rewards = [-0.02, 1, -4]
# koordynaty: 1 -- bezwzględne, 2 -- względne
cordinate = 1
# wizja: 1 -- odległosci, 2 -- obecnosci
vision = 1


print("Trwa trening na planszy", nSegments, "x", nSegments)

# ZMIENNE WYKORZYSTYWANE W PROCESIE DOTRENOWANIA
warmupGames = 0
maxMoves = 15
scoreSum = 0

# STWORZENIE ŚRODOWISKA, WCZYTANIE SIECI ORAZ DQN
env = SnakeEnvironment(waitTime=waitTime, segments=8,
                       vision=vision, livingPenalty=rewards[0],
                       posReward=rewards[1], negReward=rewards[2],
                       cordinate=cordinate, visualization=visualization)
model = load_model(pretrained_model)
DQN = Dqn(gamma, maxMemory)

# ZMIENNE WYKORZYSTYWANE W PROCESIE DOTRENOWANIA
warmupGames = 0
maxMoves = 15
scoreSum = 0

print("WARM-UP!")
# WARM-UP
while len(DQN.memory) < maxMemory:
    warmupGames += 1
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False
    while not gameOver:
        action = argmax(model.predict(currentState))
        nextState, reward, gameOver, win = env.step(action)
        DQN.remember([currentState, action, reward, nextState], gameOver)
        currentState = nextState
        if env.moves > (env.score+1)*maxMoves:
            gameOver = True
    scoreSum += env.score

warmupMeanScore = scoreSum/warmupGames
print("WARM-UP COMPLETED. PLAYED GAMES:", warmupGames,
      "MEAN SCORE:", warmupMeanScore)


print("TRAINING!")

# ZMIENNE WYKORZYSTYWANE W PROCESIE UCZENIA
epoch = 1
scoreInEpochs = []
meanScore = 0
bestMeanScore = 0
bestScore = [0, 0]
fullMemoryEpoch = 0
win = False

env = SnakeEnvironment(waitTime=waitTime, segments=nSegments,
                       vision=vision, livingPenalty=rewards[0],
                       posReward=rewards[1], negReward=rewards[2],
                       cordinate=cordinate, visualization=visualization)
model = load_model(pretrained_model)


while epoch <= epochs:
    # NOWA GRA -- reset środowiska, i początkowy input
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False

    while not gameOver:

        action = argmax(model.predict(currentState))

        # podjęcie akcji
        nextState, reward, gameOver, win = env.step(action)

        # umieszczenie ruchu w pamięci i trening sieci na pobranym batchu
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState
        if env.moves > (env.score+1)*maxMoves:
            gameOver = True

    # statystyki z pojedynczej gry
    if env.score > bestScore[0] or (env.score == bestScore[0] and
                                    env.moves < bestScore[1]):
        bestScore = [env.score, env.moves]
        print("NEW BEST SCORE:", bestScore[0], "points in",
              bestScore[1], "moves")
    print("Epoch:\t", epoch, "Score:\t", env.score, "Moves:\t", env.moves,
          "Best score:", bestScore[0])
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
        plt_name = "model_unrel_dis_warm8.jpg"
        plt.savefig(plt_name)
        plt.show()
        plt.clf()
        plt.cla()

        if meanScore/100 > bestMeanScore:
            bestMeanScore = meanScore/100
        meanScore = 0

    epoch += 1

# zapisanie modelu
model_name = "model_unrel_dis_warm8.h5"
model.save(model_name)
