# Plateforme de trading, réplication d'indice :cactus:

## Présentation du projet :racehorse:

Objectif du TD : la réplication statistique est une méthode très utilisée en finance. On l'utilise par
exemple pour déterminer la pondération de chaque sous-jacent d'un indice, pour se couvrir avec
un nombre réduit de sous-jacents, pour réduire le périmètre de recherche et de trading
(composantes principales), etc. On s’intéresse ici à la réplication de l’indice Eurostoxx 50 (50 plus
grosses capitalisations Européennes) en utilisant différentes méthodes de sélection de variables
vues en cours.

## 1. Récupération des données de l'Eurostoxx 50 dans Python.

On récupère tout d'abord les données à l'aide de la fonction "read_csv".

```
eurostoxx_df = pd.read_csv(r".\Data_Eurostoxx_10min_Feb19_Sep19.csv", sep=';', decimal=",")

Output : 
              Dates  ABI BB Equity  AD NA Equity  ADS GY Equity  AI FP Equity  ...  TEF SQ Equity  URW NA Equity  VIV FP Equity  VOW3 GY Equity  SX5E Index
0  27/02/2019 09:00          65.65        22.870          212.4        109.95  ...          7.573         143.46          24.23          150.30     3279.78
1  27/02/2019 09:10          65.69        22.490          212.2        109.90  ...          7.558         143.04          24.25          149.80     3275.70
2  27/02/2019 09:20          65.58        22.440          212.0        109.75  ...          7.552         143.00          24.14          150.30     3273.88
3  27/02/2019 09:30          65.74        22.545          211.9        109.90  ...          7.552         143.10          24.27          150.64     3277.38
4  27/02/2019 09:40          65.64        22.410          211.9        109.90  ...          7.546         142.82          24.25          150.40     3272.65

[5 rows x 50 columns]
```

On observe tout d'abord les dimensions de notre dataframe initial ainsi que les valeurs qu'il contient.
```
Output: 
# Dimensions of original data: (7011, 50)
# Number of null values: 1100
```

On supprime les lignes contenant des valeurs manquantes puis on affiche les dimensions de notre nouveau dataframe.
```
Output: 
# Dimensions of original data: (6660, 50)
```

On vérifie que notre dataframe ne contient plus de valeurs manquantes.
```
Output:
# Number of null values: 0
```

Enfin, on supprime la colonne "Dates" qui ne nous sera pas utile pour la suite.
```
eurostoxx_df_clean = eurostoxx_df_clean.drop(columns=['Dates'])
```

## Calcul des séries des rendements centrés et réduits de chaque composante et de l'Eurostoxx ('SX5E Index').

Pour ce projet, nous avons défini deux cas d'usage:
1. Le premier, faire des requêtes HTTP en utilisant l'API [Coindesk]('https://api.coindesk.com/v1/bpi/currentprice.json') pour récupérer les prix du Bitcoin en temps réel et dans différentes devises (€, $ et £). L'utilisateur a ainsi accès au prix du Bitcoin, en direct, depuis son interface lightning.
2. Créer un plugin capable d'éxecuter n'importe quel jeu python dans un autre terminal et aussi capable de récupérer le score réalisé par l'utilisateur et de l'encoder avec la fonction native "signmessage" de c-lighning.

Nos résultats : 

Le premier plugin nous permet d'afficher les valeurs du Bitcoin dans différentes devises dans le terminal où est chargé lightningd.

![usecase1](https://github.com/victorlrz/LightningPlugin/blob/main/src/btcplugin.png)

Le deuxième usecase présenté dans la vidéo ci-dessous permet d'éxécuter un jeu python puis d'encoder le score de l'utilisateur avec la fonction native "signmessage" de c-lightning.

[![usecase2](https://github.com/victorlrz/LightningPlugin/blob/main/src/hook.png)](https://www.youtube.com/watch?v=S9FJD41cBcY&feature=youtu.be)

## Overview de l'environnement :runner::dash:

Nous avons installé et configuré c-lightning sur WSL2 depuis Windows 10. Le Sous-système Windows pour Linux permet aux développeurs d’exécuter un environnement GNU/Linux (et notamment la plupart des utilitaires, applications et outils en ligne de commande) directement sur Windows, sans modification et tout en évitant la surcharge d’une machine virtuelle traditionnelle ou d’une configuration à double démarrage.

Enfin, Bitcoind et Lightningd sont configurés spécifiquement pour le testnet network.

## Overview du projet :eyes:

Nos plugins utilisent tous les deux la librairie [pylightning](https://pypi.org/project/lightning-python/). Nous importons ainsi les composants nécessaires de cette librairie pour développer nos plugins.

```from lightning import Plugin```

#### Bitcoin value plugin :chart:

Le premier plugin réalisé a pour objectif de récupérer la valeur du Bitcoin en temps réel dans différentes devises. Pour cela nous utilisons l'API mise à disposition par Coindesk. La fonction "getBTCvalue" nous permet de retourner un string contenant la valeur d'un BTC.

```
def getBTCvalue(currency):
    res = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
    resjson = res.json()
    symbol = symbols[str(currency).strip()]
    return "1 BTC = " + resjson['bpi'][currency]['rate'] + symbol
```

Cette fonction est appelée dans l'unique méthode mise à disposition à l'utilisateur. La méthode "BTCvalue" prend en paramètres deux arguments. Le premier argument "stopstr" permet à l'utilisateur de contrôler l'affichage des valeurs du BTC (afficher/arrêter l'affichage). Le deuxième paramètre "currency" permet à l'utilisateur de sélectionner dans quelle devise afficher la valeur BTC.

@plugin.method est natif à "pylightning" et permet de créer puis d'ajouter des méthodes à un plugin.

```
@plugin.method("BTCvalue")
def BTCvalue(plugin, stopstr=None, currency="USD", count=0, starttime = time.time()):
    """This plugin display BTC value for several currencies and suscribe to several RPC events"""
    global pluginRun
    
    stop = None

    if (stopstr == "True"):
        stop = True
    elif (stopstr == "False"):
        stop = False
    else:
        stop = None

    if (stop != None):
        pluginRun['running'] = stop

    if (pluginRun['running']):
        return None

    count += 1
    #plugin.log(f"tick {count}")
    plugin.log(getBTCvalue(currency))

    time.sleep(5.0 - ((time.time()) - starttime) % 5.0)
    BTCvalue(plugin, None, currency, count, starttime)
```

Enfin, @plugin.init() et @plugin.run() permettent l'initialisation de notre plugin. Pour lancer un plugin il suffit de préciser son path absolu au lancement de lightningd.

Par exemple :
> lightningd --plugin=/path/to/plugin

#### Bitcoin game emulator plugin :snake:

Ce second plugin a été plus complexe a réaliser car WSL2 ne permet pas encore la gestion de plusieurs terminaux comme le permet un environnement Ubuntu classique avec gnome-terminal ou encore xterm. D'autre part nous avons aussi du trouver un moyen pour interragir entre nos différents terminaux et récupérer le score d'un utlisateur à la fin de sa partie.

Nous avons fait nos tests sur le jeu snake. Vous trouverez le code de ce jeu dans le fichier "snake.py". Ce fichier n'a rien de particulier, c'est un snake classique. Ainsi notre plugin à la capacité de s'adapter à n'importe quel type de jeu développé en python et d'encoder son score avec la fonction native de c-lightning "signmessage".

Ce plugin est composé de deux méthodes : 
1. snake
2. getRewardMessage

La méthode snake permet de lancer le jeu snake dans un nouveau terminal. Cette méthode va ouvrir un nouveau terminal de commande qui va chercher le path du jeu en local et l'éxecuter. D'après nos tests il n'est pas possible d'éxecuter un jeu et de l'afficher directement depuis la console executant le plugin. Nous avons donc mis au point cette astuce pour executer le script désiré, l'afficher et interragir avec.

```
@plugin.method("snake")
def snake(plugin):
    """Starts a game of snake in a new console"""
    plugin.log("starting snake")

    os.system('cmd.exe /c start cmd.exe /c cmd/k python "Z:\\Bitcoin\\vic\\snake.py"')

    return None
```

La deuxième méthode, "getRewardMessage" permet de lire le flux de données sortant du jeu, de récupérer le score de l'utilisateur puis de l'encoder avec la fonction native "signmessage" de c-lightning.

```
@plugin.method("getRewardMessage")
def getRewardMessage(plugin):
    """Outputs encoded reward message to send over the lightning network after you ended your snake game"""
    file = open("/mnt/z/Bitcoin/vic/rewards/reward.txt")
    score = int(file.read())

    scorestr = f"Your score is : {score}"

    plugin.log(scorestr)

    command = f'lightning-cli signmessage "{scorestr}"'
    stream = os.popen(command)
    output = stream.read()

    jsonstr = json.loads(output)


    for key in jsonstr:
        plugin.log(f"{key}: {jsonstr[key]}")

    stream.close()
    stream, output, jsonstr = None, None, None

    return None
```

A chaque execution de ce plugin, l'utilisateur peut effectuer une partie de Snake. Son score sera affiché dans le terminal d'éxecution de lightningd et directement encodé.

![usecase1](https://github.com/victorlrz/LightningPlugin/blob/main/src/gameplugin.JPG)

## Authors :couple_with_heart: :two_men_holding_hands:
- Quentin Tourette
- Victor Larrezet
