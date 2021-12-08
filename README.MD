# TMM4135 Dimensjonering basert på elementmetoden prosjekt del 2

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