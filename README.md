# Deep Q-learing in the Snake game -- Głębokie uczenie ze wzmocnieniem w grze Snake :video_game::snake::brain:

Repozytorium zawiera kod stworzony w ramach projektu licencjackiego na kierunku studiów Kognitywistyka na Wydziale Psychologii i Kognitywistyki Uniwersytetu im. Adama Mickiewicza w Poznaniu.

Celem projektu było stworzenie i trening modelu sztucznej sieci neuronowej, zdolnej do gry w Snake'a w paradygmacie uczenia ze wzmocnieniem. Środowisko zostało oparte na zasadach kultowej gry. Modele sieci były trenowane na czterech różnych rodzajach danych wejściowych, stanowiących kombinację dwóch parametrów -- koordynat (bezwzględnych/względnych) oraz wizji (obecności/odległości). Modele były w stanie dobrze dostosować się do warunków środowiska podczas treningu na małej planszy, co ma też odzwierciedlenie w wynikach testowych. Na większych planszach nie uzyskiwały tak wysokich i zadowalających wyników, co pozostawia otwartą drogę do dalszych usprawnień i rozwoju projektu.

## Użyte paczki:
 - [pygame](https://www.pygame.org/) v. 2.0.0 :video_game:,
 - [tensorflow](https://www.tensorflow.org/) v. 1.14.0 :brain:,
 - [numpy](https://numpy.org/) v. 1.19.3 :1234:,
 - [matplotlib](https://matplotlib.org/) v. 3.3.3 :chart:.

## Repozytorium zawiera następujące pliki:
- main.py -- plik służący do treningu sieci neuronowej w środowisku Snake'a (można dobrać pożądane hiperparametry uczenia, warunki panujące w środowisku oraz sposób opisu stanów środowiska),
- neural_network.py -- plik zawierający architekturę sieci neuronowej,
- dqn.py -- plik zawierający konstrukcję pamięci agenta,
- snake_environment.py -- plik zawierający środowisko Snake'a; tu są zaimplementowany wszelkie zasady gry,
- retrain.py -- plik pozwalający dotrenować wcześniej wytrenowany model,
- test.py -- plik używany do testowania wytrenowanych modeli na różnych wielkościach plansz gry,
- snke_condaenv.yml -- plik środowiska Anaconda zawierający wszystkie potrzbene paczki w odpowiednich wersjach,
- folder trained_models zawiera wytrenowane już modele wraz z odpowiadającymi im wykresami uczenia oraz modele po próbie dotrenowania.
