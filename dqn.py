from numpy import zeros, max
from numpy.random import randint


class Dqn():

    def __init__(self, gamma, maxMemory):
        self.gamma = gamma
        self.memory = []
        self.maxMemory = maxMemory

    # dołączenie nowego przypadku do pamięci:
    # [[currentState, action, reward, nextState], gameOver]
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]

    # uzyskanie batchu do sieci z pamięci
    def getBatch(self, model, batchSize):
        nInputs = self.memory[0][0][0].shape[1]  # wymiar inputu do sieci
        nOutputs = model.output_shape[-1]  # wymiar outputu z sieci 

        # zapoczątkowanie batchu -- input i wartości docelowe
        inputs = zeros((min(batchSize, len(self.memory)), nInputs))
        targets = zeros((min(batchSize, len(self.memory)), nOutputs))

        # pobranie z pamięci losowych elementów w ilości batchSize
        randomMemory = randint(0, len(self.memory), size=min(batchSize,
                                                             len(self.memory)))
        for i in range(len(randomMemory)):
            currentState, action, reward, nextState = self.memory[
                randomMemory[i]][0]
            gameOver = self.memory[randomMemory[i]][1]
            inputs[i] = currentState
            # predykcja modelu o możliwych reward z tego currentState
            targets[i] = model.predict(currentState)[0]

            # aktualizacja targetów
            if gameOver:
                # target akcji przegrywającej się nie zmienia
                targets[i][action] = reward
            else:
                # target zostaje wzmocniony o najwyższą z możliwych
                # potencjalnych nagród z przyszłego stanu
                targets[i][action] = reward + self.gamma*max(
                    model.predict(nextState)[0])

        return inputs, targets
