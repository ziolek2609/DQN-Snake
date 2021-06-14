from random import randint
from numpy import zeros
import pygame


class SnakeEnvironment():

    # rozpoczęcie gry z ustawieniami startowymi
    def __init__(self, waitTime=100, segments=10, grid=1, segmentSize=30,
                 livingPenalty=-0.02, posReward=1, negReward=-2, vision=1,
                 cordinate=1, visualization=True):

        # STAŁE PARAMETRY ŚRODOWISKA
        # ilość segmentów na jednym boku ekranu gry
        self.SEGMENTS = segments
        # wielkość odstępu między segmentami
        self.GRID = grid
        # wielkość jednego segmentu
        self.SEGMENTSIZE = segmentSize
        # wielkość boku ekranu wyliczana na podstawie pozostałych parametrów
        self.SCREENSIZE = self.SEGMENTS * self.SEGMENTSIZE + (
            self.SEGMENTS - 1) * self.GRID
        # kara za niezjadanie jabłka
        self.LIVINGPENALTY = livingPenalty
        # nagoda za zjedzenie jabłka -- pozytywna nagroda
        self.POSREWARD = posReward
        # kara za przegraną -- negatywna nagroda
        self.NEGREWARD = negReward
        # czas pomiędzy akcjami
        self.WAITTIME = waitTime
        # wizja węża: 1 -- wizja odległości, 2 -- wizja obecności
        self.VISION = vision
        # koordynaty: 1 -- bezwzględne, 2 -- względne
        self.CORDINATE = cordinate
        # wizualizacja w pygame? -- True/False
        self.VISUALIZATION = visualization
        # ekran gry
        if self.VISUALIZATION:
            self.SCREEN = pygame.display.set_mode((
                self.SCREENSIZE, self.SCREENSIZE))

        self.reset()

    # przywrócenie zmiennych gry do ustawień początkowych
    def reset(self):
        self.direction = 1  # kierunek węża (0-up,1-right,2-down,3-left)
        self.moves = 0  # liczba wykonanych ruchów
        self.score = 0  # liczba zdobytych punktów
        self.snakeLoc = []  # lista lokalizacji segmentów węża
        # mapa ekranu: 0 -> puste pole, 0.5 -> segment snake'a, 1 -> jabłko
        self.screenMap = zeros((self.SEGMENTS, self.SEGMENTS))
        # zapoczątkowanie węża (3 segmenty) i jabłka
        for i in range(3):
            self.snakeLoc.append([2-i, 1])
            self.screenMap[1, 2-i] = 0.5
        self.appleLoc = self.createApple()
        if self.VISUALIZATION:
            self.drawScreen()

    # tworzenie jabłko w wolnej lokacji na screenMap
    def createApple(self):
        x = randint(0, self.SEGMENTS-1)
        y = randint(0, self.SEGMENTS-1)
        while self.screenMap[x][y] == 0.5 or self.screenMap[x][y] == 1:
            x = randint(0, self.SEGMENTS-1)
            y = randint(0, self.SEGMENTS-1)
        self.screenMap[x][y] = 1
        return x, y

    # rysowanie ekranu gry z elementami
    def drawScreen(self):
        self.SCREEN.fill((0, 0, 0))
        for i in range(self.SEGMENTS):
            for j in range(self.SEGMENTS):
                # segmenty węża
                if self.screenMap[j][i] == 0.5:
                    pygame.draw.rect(self.SCREEN,
                                     (0, 255, 0),
                                     (i*(self.SEGMENTSIZE+self.GRID),
                                      j*(self.SEGMENTSIZE+self.GRID),
                                      self.SEGMENTSIZE, self.SEGMENTSIZE))
                # jabłko
                elif self.screenMap[j][i] == 1:
                    pygame.draw.rect(self.SCREEN,
                                     (255, 0, 0),
                                     (i*(self.SEGMENTSIZE+self.GRID),
                                      j*(self.SEGMENTSIZE+self.GRID),
                                      self.SEGMENTSIZE, self.SEGMENTSIZE))
        pygame.display.flip()

    # pojedyncze poruszenie węża
    def moveSnake(self, nextLoc):
        # aktualizacja snakeLoc i appleLoc
        self.snakeLoc.insert(0, nextLoc)
        # niezjedzenie jabłka -> usunięcie ostatniego segmentu węża
        if nextLoc != (self.appleLoc[1], self.appleLoc[0]):
            self.snakeLoc.pop(len(self.snakeLoc)-1)
        # zjedzenie jabłka -> nowe jabłko(o ile not win) i score+1
        else:
            self.score += 1
            if self.score < self.SEGMENTS**2-3:
                self.appleLoc = self.createApple()

        # aktualizacja screenMap
        self.screenMap = zeros((self.SEGMENTS, self.SEGMENTS))
        self.screenMap[self.appleLoc[0], self.appleLoc[1]] = 1
        for i in self.snakeLoc:
            self.screenMap[i[1], i[0]] = 0.5

    # obliczenie newState, który jest inputem do sieci neuronowej
    # okreslenie stanu srodowiska
    def newState(self, wallCrush):
        # KOORDYNATY BEZWZGLĘDNE
        # input to 24 liczby: 8 kierunków (N,NE,E,SE,S,SW,W,NW)*3 informacje
        # KOORDYNATY WZGLĘDNE
        # input to 21 liczb: 7 kierunków (BL,L,FL,F,FR,R,BR)*3 informacje
        # WIZJA ODLEGŁOŚCI
        # na każdym kierunku sprawdzana odległosć do jabłka, sciany i weza
        # WIZJA OBECNOŚCI
        # na każdym kierunku sprawdzane czy jabłko i wąż są na danym kierunku,
        # 1 -- jest, 0 -- nie ma oraz odległosć do sciany
        # jeśli wąż uderzy w ściane (wallCrush==True) -- odległość od ściany
        # na danym kierunku to 0

        # BEZWZGLĘDNE -- 8 kierunków
        if self.CORDINATE == 1:
            newState = zeros((1, 24))

            # NORTH
            for i in range(0, self.snakeLoc[0][1]):
                if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                    if self.VISION == 1:
                        newState[0][0] = self.snakeLoc[0][1]-i
                    else:
                        newState[0][0] = 1
                if self.screenMap[i][self.snakeLoc[0][0]] == 0.5:
                    if self.VISION == 1:
                        newState[0][1] = self.snakeLoc[0][1]-i
                    else:
                        newState[0][1] = 0
            newState[0][2] = self.snakeLoc[0][1]+1

            # NORTH-EAST
            snakeSeen = False
            for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                            range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                if self.screenMap[i][j] == 1:
                    if self.VISION == 1:
                        newState[0][3] = j-self.snakeLoc[0][0]
                    else:
                        newState[0][3] = 1
                if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                    if self.VISION == 1:
                        newState[0][4] = j-self.snakeLoc[0][0]
                    else:
                        newState[0][4] = 1
                    snakeSeen = True
            newState[0][5] = min(self.snakeLoc[0][1]+1,
                                 self.SEGMENTS-self.snakeLoc[0][0])

            # EAST
            snakeSeen = False
            for i in range(self.snakeLoc[0][0]+1, self.SEGMENTS):
                if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                    if self.VISION == 1:
                        newState[0][6] = i-self.snakeLoc[0][0]
                    else:
                        newState[0][6] = 1
                if self.screenMap[self.snakeLoc[0][1]][i] == 0.5 and \
                        snakeSeen is False:
                    if self.VISION == 1:
                        newState[0][7] = i-self.snakeLoc[0][0]
                    else:
                        newState[0][7] = 1
                    snakeSeen = True
            newState[0][8] = self.SEGMENTS-self.snakeLoc[0][0]

            # SOUTH-EAST
            snakeSeen = False
            for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                            range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                if self.screenMap[i][j] == 1:
                    if self.VISION == 1:
                        newState[0][9] = j-self.snakeLoc[0][0]
                    else:
                        newState[0][9] = 1
                if self.screenMap[i][j] == 0.5 == 0.5 and snakeSeen is False:
                    if self.VISION == 1:
                        newState[0][10] = j-self.snakeLoc[0][0]
                    else:
                        newState[0][10] = 1
                    snakeSeen = True
            newState[0][11] = min(self.SEGMENTS-self.snakeLoc[0][0],
                                  self.SEGMENTS - self.snakeLoc[0][1])

            # SOUTH
            snakeSeen = False
            for i in range(self.snakeLoc[0][1]+1, self.SEGMENTS):
                if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                    if self.VISION == 1:
                        newState[0][12] = i-self.snakeLoc[0][1]
                    else:
                        newState[0][12] = 1
                if self.screenMap[i][self.snakeLoc[0][0]] == 0.5 and \
                        snakeSeen is False:
                    if self.VISION == 1:
                        newState[0][13] = i-self.snakeLoc[0][1]
                    else:
                        newState[0][13] = 1
                    snakeSeen = True
            newState[0][14] = self.SEGMENTS - self.snakeLoc[0][1]

            # SOUTH-WEST
            snakeSeen = False
            for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                            range(self.snakeLoc[0][0]-1, -1, -1)):
                if self.screenMap[i][j] == 1:
                    if self.VISION == 1:
                        newState[0][15] = i-self.snakeLoc[0][1]
                    else:
                        newState[0][15] = 1
                if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                    if self.VISION == 1:
                        newState[0][16] = i-self.snakeLoc[0][1]
                    else:
                        newState[0][16] = 1
                    snakeSeen = True
            newState[0][17] = min(self.snakeLoc[0][0]+1,
                                  self.SEGMENTS-self.snakeLoc[0][1])

            # WEST
            for i in range(0, self.snakeLoc[0][0]):
                if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                    if self.VISION == 1:
                        newState[0][18] = self.snakeLoc[0][0] - i
                    else:
                        newState[0][18] = 1
                if self.screenMap[self.snakeLoc[0][1]][i] == 0.5:
                    if self.VISION == 1:
                        newState[0][19] = self.snakeLoc[0][0] - i
                    else:
                        newState[0][19] = 1
            newState[0][20] = self.snakeLoc[0][0]+1

            # NORTH-WEST
            for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                            range(self.snakeLoc[0][0]-1, -1, -1)):
                if self.screenMap[i][j] == 1:
                    if self.VISION == 1:
                        newState[0][21] = self.snakeLoc[0][0] - j
                    else:
                        newState[0][21] = 1
                if self.screenMap[i][j] == 0.5:
                    if self.VISION == 1:
                        newState[0][22] = self.snakeLoc[0][0] - j
                    else:
                        newState[0][22] = 1
            newState[0][23] = min(self.snakeLoc[0][0]+1,
                                  self.snakeLoc[0][1]+1)

            if wallCrush:
                if self.direction == 0:
                    newState[0][2] = 0
                elif self.direction == 1:
                    newState[0][8] = 0
                elif self.direction == 2:
                    newState[0][14] = 0
                elif self.direction == 3:
                    newState[0][20] = 0
        
        # WZGLĘDNE -- 7 kierunków * 4 kierunki ruchu
        elif self.CORDINATE == 2:
            newState = zeros((1, 21))
            # RUCH W GÓRĘ
            if self.direction == 0:
                # BACK-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][0] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][0] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][1] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][1] = 1
                        snakeSeen = True
                newState[0][2] = min(self.snakeLoc[0][0]+1,
                                     self.SEGMENTS-self.snakeLoc[0][1])
                # LEFT
                for i in range(0, self.snakeLoc[0][0]):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][3] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][3] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5:
                        if self.VISION == 1:
                            newState[0][4] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][4] = 1
                newState[0][5] = self.snakeLoc[0][0]+1
                # FORWARD-LEFT
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][6] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][6] = 1
                    if self.screenMap[i][j] == 0.5:
                        if self.VISION == 1:
                            newState[0][7] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][7] = 1
                newState[0][8] = min(self.snakeLoc[0][0]+1,
                                     self.snakeLoc[0][1]+1)
                # FORWARD
                for i in range(0, self.snakeLoc[0][1]):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][9] = self.snakeLoc[0][1]-i
                        else:
                            newState[0][9] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5:
                        if self.VISION == 1:
                            newState[0][10] = self.snakeLoc[0][1]-i
                        else:
                            newState[0][10] = 0
                newState[0][11] = self.snakeLoc[0][1]+1
                # FORWARD-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][12] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][12] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][13] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][13] = 1
                        snakeSeen = True
                newState[0][14] = min(self.snakeLoc[0][1]+1,
                                      self.SEGMENTS-self.snakeLoc[0][0])
                # RIGHT
                snakeSeen = False
                for i in range(self.snakeLoc[0][0]+1, self.SEGMENTS):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][15] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][15] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][16] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][16] = 1
                        snakeSeen = True
                newState[0][17] = self.SEGMENTS-self.snakeLoc[0][0]
                # BACK-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][18] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][18] = 1
                    if self.screenMap[i][j] == 0.5 == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][19] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][19] = 1
                        snakeSeen = True
                newState[0][20] = min(self.SEGMENTS-self.snakeLoc[0][0],
                                      self.SEGMENTS-self.snakeLoc[0][1])
            # RUCH W PRAWO
            elif self.direction == 1:
                # BACK-LEFT
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][0] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][0] = 1
                    if self.screenMap[i][j] == 0.5:
                        if self.VISION == 1:
                            newState[0][1] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][1] = 1
                newState[0][2] = min(self.snakeLoc[0][0]+1,
                                     self.snakeLoc[0][1]+1)
                # LEFT
                for i in range(0, self.snakeLoc[0][1]):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][3] = self.snakeLoc[0][1]-i
                        else:
                            newState[0][3] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5:
                        if self.VISION == 1:
                            newState[0][4] = self.snakeLoc[0][1]-i
                        else:
                            newState[0][4] = 0
                newState[0][5] = self.snakeLoc[0][1]+1
                # FORWARD-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][6] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][6] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][7] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][7] = 1
                        snakeSeen = True
                newState[0][8] = min(self.snakeLoc[0][1]+1,
                                     self.SEGMENTS-self.snakeLoc[0][0])
                # FORWARD
                snakeSeen = False
                for i in range(self.snakeLoc[0][0]+1, self.SEGMENTS):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][9] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][9] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][10] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][10] = 1
                        snakeSeen = True
                newState[0][11] = self.SEGMENTS-self.snakeLoc[0][0]
                # FORWARD-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][12] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][12] = 1
                    if self.screenMap[i][j] == 0.5 == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][13] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][13] = 1
                        snakeSeen = True
                newState[0][14] = min(self.SEGMENTS-self.snakeLoc[0][0],
                                      self.SEGMENTS-self.snakeLoc[0][1])
                # RIGHT
                snakeSeen = False
                for i in range(self.snakeLoc[0][1]+1, self.SEGMENTS):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][15] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][15] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][16] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][16] = 1
                        snakeSeen = True
                newState[0][17] = self.SEGMENTS-self.snakeLoc[0][1]
                # BACK-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][18] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][18] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][19] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][19] = 1
                        snakeSeen = True
                newState[0][20] = min(self.snakeLoc[0][0]+1,
                                      self.SEGMENTS-self.snakeLoc[0][1])
            # RUCH W DÓŁ
            elif self.direction == 2:
                # BACK-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][0] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][0] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][1] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][1] = 1
                        snakeSeen = True
                newState[0][2] = min(self.snakeLoc[0][1]+1,
                                     self.SEGMENTS-self.snakeLoc[0][0])
                # LEFT
                snakeSeen = False
                for i in range(self.snakeLoc[0][0]+1, self.SEGMENTS):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][3] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][3] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][4] = i-self.snakeLoc[0][0]
                        else:
                            newState[0][4] = 1
                        snakeSeen = True
                newState[0][5] = self.SEGMENTS-self.snakeLoc[0][0]
                # FORWARD-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][6] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][6] = 1
                    if self.screenMap[i][j] == 0.5 == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][7] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][7] = 1
                        snakeSeen = True
                newState[0][8] = min(self.SEGMENTS-self.snakeLoc[0][0],
                                     self.SEGMENTS-self.snakeLoc[0][1])
                # FORWARD
                snakeSeen = False
                for i in range(self.snakeLoc[0][1]+1, self.SEGMENTS):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][9] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][9] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][10] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][10] = 1
                        snakeSeen = True
                newState[0][11] = self.SEGMENTS-self.snakeLoc[0][1]
                # FORWARD-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][12] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][12] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][13] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][13] = 1
                        snakeSeen = True
                newState[0][14] = min(self.snakeLoc[0][0]+1,
                                      self.SEGMENTS-self.snakeLoc[0][1])
                # RIGHT
                for i in range(0, self.snakeLoc[0][0]):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][15] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][15] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5:
                        if self.VISION == 1:
                            newState[0][16] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][16] = 1
                newState[0][17] = self.snakeLoc[0][0] + 1
                # BACK-RIGHT
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][18] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][18] = 1
                    if self.screenMap[i][j] == 0.5:
                        if self.VISION == 1:
                            newState[0][19] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][19] = 1
                newState[0][20] = min(self.snakeLoc[0][0]+1,
                                      self.snakeLoc[0][1]+1)
            # RUCH W LEWO
            elif self.direction == 3:
                # BACK-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][0] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][0] = 1
                    if self.screenMap[i][j] == 0.5 == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][1] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][1] = 1
                        snakeSeen = True
                newState[0][2] = min(self.SEGMENTS-self.snakeLoc[0][0],
                                     self.SEGMENTS - self.snakeLoc[0][1])
                # LEFT
                snakeSeen = False
                for i in range(self.snakeLoc[0][1]+1, self.SEGMENTS):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][3] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][3] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5 and \
                            snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][4] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][4] = 1
                        snakeSeen = True
                newState[0][5] = self.SEGMENTS - self.snakeLoc[0][1]
                # FORWARD-LEFT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]+1, self.SEGMENTS),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][6] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][6] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][7] = i-self.snakeLoc[0][1]
                        else:
                            newState[0][7] = 1
                        snakeSeen = True
                newState[0][8] = min(self.snakeLoc[0][0]+1,
                                     self.SEGMENTS-self.snakeLoc[0][1])
                # FORWARD
                for i in range(0, self.snakeLoc[0][0]):
                    if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                        if self.VISION == 1:
                            newState[0][9] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][9] = 1
                    if self.screenMap[self.snakeLoc[0][1]][i] == 0.5:
                        if self.VISION == 1:
                            newState[0][10] = self.snakeLoc[0][0] - i
                        else:
                            newState[0][10] = 1
                newState[0][11] = self.snakeLoc[0][0] + 1
                # FORWARD-RIGHT
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]-1, -1, -1)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][12] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][12] = 1
                    if self.screenMap[i][j] == 0.5:
                        if self.VISION == 1:
                            newState[0][13] = self.snakeLoc[0][0] - j
                        else:
                            newState[0][13] = 1
                newState[0][14] = min(self.snakeLoc[0][0]+1,
                                      self.snakeLoc[0][1]+1)
                # RIGHT
                for i in range(0, self.snakeLoc[0][1]):
                    if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                        if self.VISION == 1:
                            newState[0][15] = self.snakeLoc[0][1] - i
                        else:
                            newState[0][15] = 1
                    if self.screenMap[i][self.snakeLoc[0][0]] == 0.5:
                        if self.VISION == 1:
                            newState[0][16] = self.snakeLoc[0][1] - i
                        else:
                            newState[0][16] = 0
                newState[0][17] = self.snakeLoc[0][1] + 1
                # BACK-RIGHT
                snakeSeen = False
                for i, j in zip(range(self.snakeLoc[0][1]-1, -1, -1),
                                range(self.snakeLoc[0][0]+1, self.SEGMENTS)):
                    if self.screenMap[i][j] == 1:
                        if self.VISION == 1:
                            newState[0][18] = j-self.snakeLoc[0][0]
                        else:
                            newState[0][18] = 1
                    if self.screenMap[i][j] == 0.5 and snakeSeen is False:
                        if self.VISION == 1:
                            newState[0][19] = j - self.snakeLoc[0][0]
                        else:
                            newState[0][19] = 1
                        snakeSeen = True
                newState[0][20] = min(self.snakeLoc[0][1]+1,
                                      self.SEGMENTS-self.snakeLoc[0][0])

            # ZDERZENIE ZE ŚCIANĄ
            if wallCrush:
                newState[0][11] = 0

        return newState

    # podjęcie akcji -- aktualizacja stanu gry
    def step(self, action):
        self.moves += 1
        gameOver = False
        wallCrush = False
        win = False
        reward = self.LIVINGPENALTY  # domyślna nagroda to livingPenalty

        if self.CORDINATE == 2:
            # krok w grze -- możliwe scenariusze --
            # ustalenie nagrody i stanów: gameOver, win oraz wallCrush

            # RUCH W GÓRĘ
            if (action == 0 and self.direction == 1) \
                    or (action == 1 and self.direction == 0) \
                    or (action == 2 and self.direction == 3):
                if self.snakeLoc[0][1] > 0:
                    # zjedzenie samego siebie -> PRZEGRANA
                    if self.screenMap[
                            self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    # zjedzenie jabłka -> POZYTYWNA NAGRODA
                    elif self.screenMap[
                            self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 1:
                        reward = self.POSREWARD
                    # przeunięcie węża
                    self.moveSnake((self.snakeLoc[0][0],
                                    self.snakeLoc[0][1]-1))
                # wyjście poza mapę -> PRZEGRANA
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD
                self.direction = 0
            # RUCH W PRAWO
            elif (action == 0 and self.direction == 2) \
                    or (action == 1 and self.direction == 1) \
                    or (action == 2 and self.direction == 0):
                if self.snakeLoc[0][0] < self.SEGMENTS-1:
                    if self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0]+1,
                                    self.snakeLoc[0][1]))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD
                self.direction = 1
            # RUCH W DÓŁ
            elif (action == 0 and self.direction == 3) \
                    or (action == 1 and self.direction == 2) \
                    or (action == 2 and self.direction == 1):
                if self.snakeLoc[0][1] < self.SEGMENTS-1:
                    if self.screenMap[
                            self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0],
                                    self.snakeLoc[0][1]+1))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD
                self.direction = 2
            # RUCH W LEWO
            elif (action == 0 and self.direction == 0) \
                    or (action == 1 and self.direction == 3) \
                    or (action == 2 and self.direction == 2):
                if self.snakeLoc[0][0] > 0:
                    if self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0]-1,
                                    self.snakeLoc[0][1]))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD
                self.direction = 3

        elif self.CORDINATE == 1:
            # wykluczneie możliwości skrętu o 180 stopni
            if action == 1 and self.direction == 3:
                action = 3
            elif action == 2 and self.direction == 0:
                action = 0
            elif action == 3 and self.direction == 1:
                action = 1
            elif action == 0 and self.direction == 2:
                action = 2

            # ustalenie nagrody i stanów: gameOver, win oraz wallCrush
            if action == 0:  # ruch w górę
                if self.snakeLoc[0][1] > 0:
                    # zjedzenie samego siebie -> PRZEGRANA
                    if self.screenMap[
                            self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    # zjedzenie jabłka -> POZYTYWNA NAGRODA
                    elif self.screenMap[
                            self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0],
                                    self.snakeLoc[0][1]-1))
                else:  # wyjście poza mapę -> PRZEGRANA
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD

            elif action == 1:  # ruch w prawo
                if self.snakeLoc[0][0] < self.SEGMENTS-1:
                    if self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0]+1,
                                    self.snakeLoc[0][1]))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD

            elif action == 2:  # ruch w dół
                if self.snakeLoc[0][1] < self.SEGMENTS-1:
                    if self.screenMap[
                            self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0],
                                    self.snakeLoc[0][1]+1))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD

            elif action == 3:  # ruch w lewo
                if self.snakeLoc[0][0] > 0:
                    if self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 0.5:
                        gameOver = True
                        reward = self.NEGREWARD
                    elif self.screenMap[
                            self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 1:
                        reward = self.POSREWARD
                    self.moveSnake((self.snakeLoc[0][0]-1,
                                    self.snakeLoc[0][1]))
                else:
                    gameOver = True
                    wallCrush = True
                    reward = self.NEGREWARD
            self.direction = action

        # WYGRANA!:)
        if self.score == self.SEGMENTS**2-3:
            win = True
            gameOver = True
            print("YOU ARE THE WINNER!")

        # aktualizacja stanu gry(inputu do sieci) i ekranu
        if self.VISUALIZATION:
            self.drawScreen()
            pygame.time.wait(self.WAITTIME)
        newState = self.newState(wallCrush)

        return newState, reward, gameOver, win

# główna pętla do gry samodzielnej
if __name__ == '__main__':
    env = SnakeEnvironment(waitTime=1000, vision=1, cordinate=2)
    action = 2
    win = False
    gameOver = False
    while not win:
        # sterowanie
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 2
                    _, _, gameOver, win = env.step(action)
                elif event.key == pygame.K_RIGHT:
                    action = 3
                    _, _, gameOver, win = env.step(action)
                elif event.key == pygame.K_LEFT:
                    action = 1
                    _, _, gameOver, win = env.step(action)

        # podjęcie akcji (przy grze samodzielnej potrzeba tylko gameOver i win;
        # reward i newState używa się w treningu)
        _, _, gameOver, win = env.step(action)

        if gameOver:  # przegrana i wyświetlenie wynikóww konsoli
            print("WYNIK: ", env.score,"\nLICZBA RUCHÓW: ", env.moves)
            # reset środowiska
            env.reset()
            action = 2
            gameOver = False
