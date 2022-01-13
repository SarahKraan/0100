# Eindrapport 
## Inhoudsopgave
[1: Basismodel](#1-basismodel)\
2:...\
3: ...\
4: ...\
5: ...\
6: ...\
7: ...\
8: ...\
9: ...\
10: ...

## 1. Basismodel

### Introductie
Het aantal bedreigde diersoorten neemt elk jaar toe. Dit is te wijten aan onder andere de afnemende grootte van leefruimte voor dieren, het wegvallen van primaire of non-primaire voedselbronnen, een toenemend aantal stroperijen en klimaatveranderingen. Het is cruciaal om kennis te hebben van de diersoorten die in de nabije toekomst een groot risico lopen om een bedreigde diersoort te worden. Een hulpmiddel dat voor dit doeleinde kan worden gebruikt is het tellen van het aantal dieren dat van een bepaalde soort bestaat. Hierdoor kan in kaart worden gebracht of het aantal levende dieren van een bepaalde soort wellicht te laag is en er dus om actie wordt gevraagd om dit aantal omhoog te brengen door middel van bijvoorbeeld extra bescherming en het stimuleren van voortplanting. Echter, het proces van tellen is een enorm langdradig en routineus proces dat zich perfect zou lenen voor een meer geautomatiseerde vervanging in de vorm van een systeem dat gebruikt maakt van kunstmatige intelligentie (KI). Dit geautomatiseerde proces zou tot minder nodige mankracht én minder fouten kunnen leiden, waardoor er meer tijd en energie overblijft voor andere nuttige werkzaamheden.

Het netwerk in het huidige project voorspelt tot welke soort de vogel op de afbeelding die aan het netwerk wordt meegegeven behoort. Er is hierbij een mogelijke uitkomst van één van 325 verschillende vogelsoorten. 

Om het netwerk te trainen zijn 47.000 afbeeldingen van 325 verschillende soorten vogels gebruikt. Elke soort vogel betreft minstens 120 afbeeldingen, zowel de mannelijke als de vrouwelijke variant. Mannelijke vogelsoorten zijn over het algemeen kleurrijker dan vrouwelijke vogelsoorten. De afbeeldingen gebruikt voor het trainen van het netwerk bestaan voornamelijk uit mannelijke vogelsoorten, namelijk voor 85%. Dit zou kunnen leiden tot een verminderde nauwkeurigheid voor het voorspellen van vrouwelijke vogelsoorten. Echter, de grote hoeveelheid aan afbeeldingen in de gebruikte dataset en de consistente afmetingen van de afbeeldingen maken de huidige dataset een weloverwogen passende keuze om het netwerk te trainen. 

Het huidige project helpt mee aan het immense probleem omtrent het toenemende aantal van bedreigde diersoorten op meerdere manieren. Als eerste biedt het ontwikkelde netwerk een manier om een afnemend aantal in bepaalde vogelsoorten vroegtijdig waar te kunnen nemen. Daarnaast doet het huidige netwerk dit op een manier waarbij minder beroep op de mens wordt gedaan en waarbij het tellen van vogels zowel sneller als met minder fouten kan plaatsvinden. Al met al belooft het huidige project een grote bijdrage aan het beschermen van een groot aantal vogelsoorten tegen uitsterven. 

#### Specifieke probleem
Het specifieke probleem dat in dit gedeelte van het eindrapport wordt behandeld is het in kaart brengen van de beschikbare data en haar eigenschappen. Verder is er een eerste model van het netwerk ontwikkeld dat als basis dient voor de komende versies en aanpassingen van het netwerk in de hoofdstukken die volgen.

#### Overzicht model
Voor de opbouw van het basismodel is gebruik gemaakt van de TensorFlow bibliotheek. Om te beginnen zijn een aantal basislagen aan het model toegevoegd. Hierbij zijn  om en om twee Conv2D lagen en MaxPooling lagen gebruikt, gevolgd door een Flatten en Dense laag. 

### Data Analyse en Voorverwerking
De dataset 325 Birds Species te vinden op kaggle (hier linkje van maken) voorziet ons van alle data die nodig is om dit onderzoek uit te voeren. Deze dataset bestaat uit 50582 foto’s in totaal van 325 soorten vogels. De afbeeldingen hebben als afmeting allemaal 224 (pixels) x 224 (pixels) x 3 (lagen). De drie lagen houden in dat de afbeeldingen volgens het RGB-systeem gekleurd zijn. Hieronder volgen een aantal voorbeeldafbeeldingen uit de dataset: 

Binnen de dataset was al onderscheid gemaakt tussen training-, test- en validatie data; de training data bevat 47332 afbeeldingen en de test- en validatie data allebei 1625 afbeeldingen. Deze verdeling is zo optimaal mogelijk gedaan om een zo precies mogelijke voorspelling over de vogelsoort bij een afbeelding te kunnen maken. Vanwege deze verdeling die al in de dataset aanwezig was, is het niet nodig om een eigen onderverdeling te maken. 

Als eerste werd alle data ingeladen. Om verder met de data te kunnen werken was het nodig om het formaat van de afbeeldingen te wijzigen. Wanneer de oorspronkelijke afmetingen van de afbeeldingen aan werden gehouden, crashte namelijk de server die werd gebruikt voor het huidige project in verband met een overbelast RAM geheugen. Daarom zijn de afbeeldingen verkleind naar afmeting 85 x 85 pixels; dit was de grootste afmeting van de afbeeldingen waarbij de server die werd gebruikt niet crashte.
Ook zijn de labels van de afbeeldingen omgezet in een matrixvorm door zogeheten ‘one-hot encoding’, waarbij een 1 wordt geplaatst bij het label van de juiste categorie vogel en bij de andere categorieën een 0. Op deze manier kan de matrix van labels vergeleken worden met de uitkomst van het model.


### Model Pipeline en Training
Het basismodel neemt als input de training data die bestaat uit de 47332 training afbeeldingen verwerkt in een matrix van afmeting 47332 x 85 x 85  x 3 en de bijbehorende matrix van training labels met afmeting 47332 x 1. Deze afbeeldingen worden gevoed aan de eerste laag van het model, een Conv2D-laag die 32 filters bevat. Door middel van ReLu activatie worden deze nodes wel of niet geactiveerd. Vervolgens wordt het formaat van de afbeeldingen teruggebracht naar een kleiner formaat door pooling per vier pixels toe te passen. Alleen de maximale waarde van zo’n cluster van vier pixels wordt dan meegenomen. Vervolgens gebeurt nogmaals hetzelfde met een Conv2D-laag van 64 filters en opnieuw pooling per vier pixels. 
Ten slotte worden de pixels door middel van flatten weer bij elkaar gevoegd, en kan de foto geclassificeerd worden onder een van de klassen door middel van een sigmoid activatie. De sigmoid activatie is het beste te gebruiken voor het classificeren omdat het een kans kan berekenen voor elke uitkomst. Het model pakt dan de kansen en geeft als voorspelling de klasse met de hoogste kans terug. Het aantal noden van de laatste laag is dus ook gelijk aan het aantal categorieën vogelsoorten, 325. 

Voor het trainen van het model wordt een aantal van 20 epochs aangehouden. Dit aantal is gekozen omdat dit voldoende mogelijkheden voor het model oplevert om te trainen, zonder onnodig veel verwerkingstijd te eisen zoals bij een hoger aantal epochs het geval is.


### Evaluatie en Conclusies
Evaluatie door analyseren van de training en validatie resultaten:
Na een aantal keer het model gerund te hebben is te zien dat het model overfit (zoals te zien is in de afbeelding hieronder). Dit is makkelijk te herkennen aan de trainingskosten die vrijwel op 0 zit na de 20 epochs terwijl de validatie kosten omhoog blijft gaan. Ook is te zien dat de accuraatheid van de validatie data blijft steken tussen de 15 en 20% terwijl de trainingsdata een accuraatheid heeft van 95%. 

![image](https://user-images.githubusercontent.com/68432564/149306288-c19f45f4-3298-4399-aff4-fe37e739d33e.png)

Het huidige model dient als basismodel om vanuit verder te werken en is nog niet voldoende functioneel. Het classificeren van de verschillende vogelsoorten kan namelijk nog niet optimaal. De volgende stap is daarom om het model te optimaliseren.

Om ervoor te zorgen dat het model niet meer overfit een aantal verschillende mogelijkheden worden toepassen. Er zou bijvoorbeeld een cross-validation kunnen worden toegevoegd om te zorgen dat het model niet meer wordt getraind op 47.000 afbeeldingen maar op splits van de data. Ook kan er worden gestopt met trainen wanneer de training data cost blijft dalen, maar de validatie data cost stijgt. Dit is namelijk het punt waar het model begint met overfitten. Wanneer het model op dit moment stopt met trainen kan het overfitten worden voorkomen. Het overfitten kan ook worden tegengegaan door middel van batch normalization, waarbij het model robuuster wordt gemaakt voor variërende input. Verder kunnen er nog een aantal veranderingen worden doorgevoerd om te testen of het model hier accurater van wordt. Zo kan er worden gekeken naar het eventueel toevoegen van andere activaties. Het model is nu getraind met ReLU activaties voor de hidden layers en een sigmoid activatie als laatst. Deze activatie kan worden aangepast naar verschillende activaties, bv.: softmax, tanh of ReLU. 


## HOOFDSTUK 2 EN ALLE VOLGENDE HOOFDSTUKKEN

### Introductie
Specifieke probleem die deze versie probeert aan te pakken:
High level overview van veranderingen van dit model t.o.v. het vorige model:

### Data Analyse en Voorverwerking
Mogelijke data analyse of aanpassingen in de data voor het runnen van nieuwe model:

### Model Pipeline en Training
Welke input begint het model mee:
Wat voor type model is gebruikt (inclusief welke layers van welke grootte + welke activatie functie(s):
Post processing steps vóór het maken van de voorspelling:
Batch size / number of epochs:

### Evaluatie en Conclusies
Evaluatie door analyseren van de training en validatie resultaten:
Vergelijken met het vorige model:
Invoegen van visualisaties van het trainingsproces:
Mogelijke trade-offs in vergelijking met vorige model (wat nu wellicht slechter gaat):
Heeft dit model het probleem uit de introductie verholpen:
Analyseren van mogelijke verdere verbeteringen (voor volgende hoofdstukken):





