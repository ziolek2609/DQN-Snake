from snake_environment import SnakeEnvironment
from numpy import argmax
from keras.models import load_model


# USTAWIENIA ŚRODOWISKA
# czas pomiędzy przesunięciami węża
waitTime = 0
# ilość segmentów na jednym boku ekranu
nSegments = 8
# wizja: 1 -- odległosci, 2 -- obecnosci
vision = 1
# koordynaty: 1 -- bezwzględne, 2 -- względne
cordinate = 2
# czy pokazywać wizualizacje w pygame?
visualization = False
# ilosc prob testowych
nAttempts = 5000
# nazwa modelu do testu
model_name = "model.h5"

# ZMIENNE UŻYWANE PODCZAS TESTOWANIA
maxMoves = 50
means = []
winNum = 0
means = []
best = 0
scoreSum = 0

# WCZYTANIE MODELU I ŚRODOWISKA
env = SnakeEnvironment(visualization=visualization, vision=vision,
                       segments=nSegments, waitTime=waitTime,
                       segmentSize=40, cordinate=cordinate)
model = load_model(model_name)


print(nSegments, "x", nSegments)
for i in range(nAttempts):
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False
    while not gameOver:
        action = argmax(model.predict(currentState))
        nextState, reward, gameOver, win = env.step(action)
        currentState = nextState
        if env.moves > (env.score+1)*maxMoves:
            gameOver = True
        if env.score > best:
            best = env.score
        if win:
            winNum += 1
    scoreSum += env.score
    print("Score:\t", env.score, "Moves:\t", env.moves)
modelMean = scoreSum/nAttempts
means.append(modelMean)

print("Mean score for model", model_name, ":", nAttempts, "attempts:",
      scoreSum/nAttempts, ", best:", best, ", wins:", winNum)
