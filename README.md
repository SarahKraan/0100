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
De code van dit model is geschreven in Python 3.6.9. De vereiste bibliotheken en/of programma's die nodig zijn om de code van het huidige project te runnen zijn:
* import numpy as np
* import cv2
* import os
* from PIL import Image
* import pandas as pd
* import tensorflow as tf
* import matplotlib.pyplot as plt
* from tensorflow.keras import layers, models, preprocessing
* from tensorflow.keras.utils import to_categorical
* from sklearn.preprocessing import LabelEncoder

## Gebruiksaanwijzing
Instructies om het netwerk model te gebruiken zijn als volgt:
1. Download de '325 Bird Species - Classification' dataset op uw computer via de link: https://www.kaggle.com/gpiosenka/100-bird-species
2. Voeg aan uw Google Drive een map toe genaamd: 'AIproject'
3. In de map 'AIproject', upload de '325 Bird Species - Classification' dataset
4. Open het model via de map '/code/basismodel' in de repository, en voer het programma uit.

## Auteurs
* Sarah Kraan
* Laurens de Jonge
* Tanja Milos
* Merlijn Däscher
