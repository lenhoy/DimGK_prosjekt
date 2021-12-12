# TMM4135 Dimensjonering basert på elementmetoden prosjekt del 2


### Note om vis
Det er gjort endringer av vis.py, som er en del av calfem repo, filen med endringer ligger under. Endringene går ut på "is not"->"!=" siden dette eller gir problemer i python >3.8.

I tillegg er det lagt til verdier for element type 9, 10 (tri6 og quad9) i: 
 - fPerElms
 - _preMeshDrawPrep()


## Installasjon
Dersom du ikke har pipenv:  
```terminal
pip install pipenv 
```
Derretter gå til mappen med pipfile og kjør:  
```
pipenv install
```  
Dette vil installere numpy, scipy, calfem-python og gmsh i et
virtuelt miljø.  

Når dette er ferdig, kjør 
```
pipenv shell
```  
i prosjektmappen for 
å starte opp det virtuelle miljøet. 
Husk å velge det virtuelle miljøet som python interpreter i 
programmet du bruker. 

Dersom det virtuelle miljøet ikke dukker opp i listen bruk  
**`pipenv --venv`** til å printe ut plasseringen til miljøet i
terminal.