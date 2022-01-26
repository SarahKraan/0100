# Birds

## Inhoudsopgave

* Introductie
* Structuur
* Vereisten
* Gebruiksaanwijzing

## Introductie
In het huidige project is een Convolutional Neural Network (CNN) ontwikkeld dat afbeeldingen van vogels over 325 verschillende vogelsoorten kan classificeren.
Er zijn meer dan 47.000 gekleurde (RGB) afbeeldingen gebruikt om het netwerk te trainen en de afmetingen van deze afbeeldingen zijn 224 x 224 x 3. De afbeeldingen die zijn gebruikt voor het trainen van het netwerk bestaan uit minstens 120 afbeeldingen per vogelsoort, bestaande uit zowel vrouwelijke als mannelijke varianten van elke soort, waarbij ongeveer 85% van alle afbeeldingen vogels van het mannelijke geslacht weergeven. 

## Structuur
Het huidige project bestaat uit de volgende onderdelen te vinden in deze repository:
* /docs: Bevat alle documenten van het huidige project.
  * /docs/eindrapport: Bevat het eindrapport met beschrijvingen van alle versies van het gemaakte netwerk inclusief toelichtingen en motivatie voor de gemaakte keuzes.
  * /docs/afbeeldingen: Bevat alle afbeeldingen die zijn gebruikt in het eindrapport (per hoofdstuk)
* /code: Bevat alle code van het huidige project.
  * /code/basismodel: Bevat een Google Colab link naar de code voor het basismodel.

## Vereisten
De code van dit model is geschreven in Python 3.6.9. De vereiste packages die nodig zijn om de code van het huidige project uit te voeren worden weergeven in de requirements.txt. Om de packages te installeren kan gebruik worden gemaakt van pip, door middel van het uitvoeren van de hieronder weergeven instructie:

```
pip install -r requirements.txt
```


## Gebruiksaanwijzing
Om het model uit te voeren, moet de data van de afbeeldingen van de vogels beschikbaar worden gemaakt. Dit kan op 2 manieren:
1. Door middel van de data te downloaden en toe te voegen aan uw Google Drive. Hiervoor moet de data gedownload worden van de '325 Bird Species - Classification' dataset op uw computer via de link: https://www.kaggle.com/gpiosenka/100-bird-species. Vervolgens voegt u de gedownloade data toe aan uw Google Drive onder een map genaamd: 'AIproject'. In de map 'AIproject' upload u de '325 Bird Species - Classification' dataset.
2. Door middel van het gebruik te maken van uw API token. Hiervoor gaat u naar de website www.kaggle.com en vervolgens naar uw account. Hier klikt u op 'Create New API token'. Er wordt een kaggle.json bestand gedownload op uw computer. Dit json bestand kunt u vervolgens uploaden wanneer u de code van het model runt.

Nadat u de data hebt ingeladen, opent u het model via de map '/code/basismodel' in de repository, en voert u het programma uit.


## Auteurs
* Sarah Kraan
* Laurens de Jonge
* Tanja Milos
* Merlijn Däscher
