# TMM4135 Dimensjonering basert på elementmetoden prosjekt del 2

## Installasjon
Kopier filene i mappen "calfem" fra handout zip filen til lokal installasjon av calfem (i virtual env/Lib/site-packages)


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

## Data Eksport
`cantilever_calc_loop.py` bygger opp et dictionary med endeforskyvninger og tilhørende noder mm. Dette gjøres om til et pandas dataframe og videre til csv og excel filer som lagres.  
Vi brukte Microsoft SandDance til å visualisere dataen og det er lagret et par "snapshots" i sanddance mappen her. Gå til [SandDance]([https://link](https://microsoft.github.io/SandDance/app/)) for å laste opp `exportdata_x40.csv` og `exportdata.snapshots` filene for å interaktivt se på dataen.