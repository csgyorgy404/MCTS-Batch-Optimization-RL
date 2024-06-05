## Monte Carlo Tree Search (MCTS)

1. **MCTS:**
    - **a)** Begyüjteszt random lépésekkel 10k Training samplet (S,A,R,S',D)
    - **b)** Meghatározod, hogy kettő a branching factor
    - **c)** Random samplingelsz uniform módon batch size-nyi training samplet amikor egy action-t választasz.
    - **d)** Befitteled a batch-t
    - **e)** Attol függően hanyadik rétegben vagy a fában (tanitás eredeti epizód száma - mar befittelt batch-k száma) maradék számú batch-t random kivalasztod és befitteled
    - **f)** 1x kiértékeled a modeltt (inditassz egy epoch-t megnezed mennyi rewardot kapott) ez a reward lesz a rollout kiindulási node-janak eredeti értéke, illetve ezt a reward erteket propagalod fel a faban.
    - **g)** Ezt az eredeti node erteket megörzöd külön, illetve letrehozol egy valtozod amiben majd kumulalod is a felpropagalt értékeket az uct-hez, az előbbi a konvergencia plot kirajzolasahoz kell majd az utóbbi az mcts működéséhez.
    - **h)** Ezt megcsinalod annyiszor ahany epizódot akartal eredetileg tanítani.
    - **j)** (Fun fact: lehetne olyan kilépési feltételt irni az mcts iteralasahoz, hogy amikor a node létrejön mert befittelted a batch-t akkor futtatsz egy kiértékelést -nem rollout-t és ha eléri azt a pontszamot ami a betanult allapothoz tartozik akkor befejezet az mcts iteralasat mert betanítottad a modelt)

2. **MCTS:**
    - **a)** Ugyan az mint az előző addig a pontig amig a rollout-hoz jutsz, ahol nem rollout-t hasznalsz valu assignment módszerként az adott node esetében, hanem egy egyszerű kiértékelést, futatsz (egy sima epizód lejatszasa a környezettel és megnézzük, hogy mennyi rewardot kap és ez lesz a node erteke, de ugyan ugy kettő valtozóba kell tenni, egy a kumulalshoz kell majd az uct miatt egy pedig a konvergencia kiralyzolasahoz.

3. **MCTS:**
    - **a)** Vizualizaciós feature implementalasa, ami az adott iteracióban kiralyzolja a best path-t a faban ezt a best path ugynezki h az y tengely az adott nodehoz tartozó modell kiértékelése soran kiszamolt pontérték az x tengely pedig az adott mélységet jelenti a faban ez a melyseg azonos az adott utvonalon befittelt batchek szamaval. ezen keresztül iteració közben lathatnank hogy tanul a modell. vagy csak bizonyos iteraciónként belenézhetnénk, hol jar.
