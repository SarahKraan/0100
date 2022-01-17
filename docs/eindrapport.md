# Eindrapport 
## Inhoudsopgave
[1: Inleiding](#1-inleiding)\
[2: Basismodel](#2-basismodel)\
[3: Drop-out methode tegen overfitten](#3-drop---out-methode-tegen-overfitten)\
[4: Verdiepen van het netwerk](#4-verdiepen-van-het-netwerk)\
5: ...\
6: ...\
7: ...\
8: ...\
9: ...\
10: …

## 1. Inleiding
Het aantal bedreigde diersoorten neemt elk jaar toe. Dit is te wijten aan onder andere de afnemende grootte van leefruimte voor dieren, het wegvallen van primaire of non-primaire voedselbronnen, een toenemend aantal stroperijen en klimaatveranderingen. Het is cruciaal om kennis te hebben van de diersoorten die in de nabije toekomst een groot risico lopen om een bedreigde diersoort te worden. Een hulpmiddel dat voor dit doeleinde kan worden gebruikt is het tellen van het aantal dieren dat van een bepaalde soort bestaat. Hierdoor kan in kaart worden gebracht of het aantal levende dieren van een bepaalde soort wellicht te laag is en er dus om actie wordt gevraagd om dit aantal omhoog te brengen door middel van bijvoorbeeld extra bescherming en het stimuleren van voortplanting. Echter, het proces van tellen is een enorm langdradig en routineus proces dat zich perfect zou lenen voor een meer geautomatiseerde vervanging in de vorm van een systeem dat gebruikt maakt van kunstmatige intelligentie (KI). Dit geautomatiseerde proces zou tot minder nodige mankracht én minder fouten kunnen leiden, waardoor er meer tijd en energie overblijft voor andere nuttige werkzaamheden.

## 2. Basismodel

### Introductie
Het doel van het netwerk in het huidige project is om te voorspellen tot welke soort de vogel op een afbeelding die aan het netwerk wordt meegegeven behoort. Er is hierbij een mogelijke uitkomst van één van 325 verschillende vogelsoorten. 

Om het netwerk te trainen zijn 47.000 afbeeldingen van 325 verschillende soorten vogels gebruikt. Elke soort vogel betreft minstens 120 afbeeldingen, zowel de mannelijke als de vrouwelijke variant. Mannelijke vogelsoorten zijn over het algemeen kleurrijker dan vrouwelijke vogelsoorten. De afbeeldingen gebruikt voor het trainen van het netwerk bestaan voornamelijk uit mannelijke vogelsoorten, namelijk voor 85%. Dit zou kunnen leiden tot een verminderde nauwkeurigheid voor het voorspellen van vrouwelijke vogelsoorten. Echter, de grote hoeveelheid aan afbeeldingen in de gebruikte dataset en de consistente afmetingen van de afbeeldingen maken de huidige dataset een weloverwogen passende keuze om het netwerk te trainen. 

Het huidige project helpt mee aan het immense probleem omtrent het toenemende aantal van bedreigde diersoorten op meerdere manieren. Als eerste biedt het ontwikkelde netwerk een manier om een afnemend aantal in bepaalde vogelsoorten vroegtijdig waar te kunnen nemen. Daarnaast doet het huidige netwerk dit op een manier waarbij minder beroep op de mens wordt gedaan en waarbij het tellen van vogels zowel sneller als met minder fouten kan plaatsvinden. Al met al belooft het huidige project een grote bijdrage aan het beschermen van een groot aantal vogelsoorten tegen uitsterven. 

#### Specifieke probleem
Het specifieke probleem dat in dit gedeelte van het eindrapport wordt behandeld is het in kaart brengen van de beschikbare data en haar eigenschappen. Verder is er een eerste model van het netwerk ontwikkeld dat als basis dient voor de komende versies en aanpassingen van het netwerk in de hoofdstukken die volgen.

#### Overzicht model
Voor de opbouw van het basismodel is gebruik gemaakt van de TensorFlow bibliotheek. Om te beginnen zijn een aantal basislagen aan het model toegevoegd. Hierbij zijn om en om twee Conv2D lagen en MaxPooling lagen gebruikt, gevolgd door een Flatten en Dense laag.

### Data Analyse en Voorverwerking
De dataset 325 Birds Species te vinden op kaggle (hier linkje van maken) voorziet ons van alle data die nodig is om dit onderzoek uit te voeren. Deze dataset bestaat uit 50582 foto’s in totaal van 325 soorten vogels. De afbeeldingen hebben als afmeting allemaal 224 (pixels) x 224 (pixels) x 3 (lagen). De drie lagen houden in dat de afbeeldingen volgens het RGB-systeem gekleurd zijn. Hieronder volgen een aantal voorbeeldafbeeldingen uit de dataset: 

Binnen de dataset was al onderscheid gemaakt tussen training-, test- en validatiedata; de trainingsdata bevat 47332 afbeeldingen en de test- en validatiedata allebei 1625 afbeeldingen. Deze verdeling is zo optimaal mogelijk gedaan om een zo precies mogelijke voorspelling over de vogelsoort bij een afbeelding te kunnen maken. Vanwege deze verdeling die al in de dataset aanwezig was, is het niet nodig om een eigen onderverdeling te maken. 

Als eerste werd alle data ingeladen. Om verder met de data te kunnen werken was het nodig om het formaat van de afbeeldingen te wijzigen. Wanneer de oorspronkelijke afmetingen van de afbeeldingen aan werden gehouden, crashte namelijk de server die werd gebruikt voor het huidige project in verband met een overbelast RAM geheugen. Daarom zijn de afbeeldingen verkleind naar afmeting 85 x 85 pixels; dit was de grootste afmeting van de afbeeldingen waarbij de server die werd gebruikt niet crashte.

Naast dat de afbeeldingen verkleind worden, worden ze ook genormaliseerd aan de hand van de ImageDataGenerator preprocesser tool (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) . Dit wordt gedaan omdat de lagen in het model gebruik maken van ReLu activatie functies. ReLu activatie functies zijn niet-lineair als de input ervan 0 is. Bij dit Neuraal Netwerk is het van belang dat de lagen niet-lineair zijn. Om er dus voor te zorgen dat de ReLu activatie functies effectief zijn, moet de input data een gemiddelde van 0 hebben. Dit wordt dus bereikt door de afbeeldingen te normaliseren naar een gemiddelde van 0 en een standaarddeviatie van 1. Door de data zo te preprocessen, zullen de afbeeldingen meer gelijk aan elkaar zijn. Dit houdt in dat bijvoorbeeld een vaag gekleurde afbeelding met lage pixels evenveel meegerekend wordt als een afbeeldingen met hele felle kleuren. Verder helpt data preprocessen het model sneller te maken.

Ook zijn de labels van de afbeeldingen omgezet in een matrixvorm door zogeheten ‘one-hot encoding’, waarbij een 1 wordt geplaatst bij het label van de juiste categorie vogel en bij de andere categorieën een 0. Op deze manier kan de matrix van labels vergeleken worden met de uitkomst van het model.


### Model Pipeline en Training
Het basismodel neemt als input de trainingsdata die bestaat uit 47332 afbeeldingen verwerkt in een matrix van afmeting 47332 x 85 x 85  x 3 en de bijbehorende matrix van labels met afmeting 47332 x 1. Deze afbeeldingen worden gevoed aan de eerste laag van het model, een Conv2D-laag die 32 filters bevat. Per afbeelding wordt padding met specificatie ‘same’ toegevoegd, waarbij er pixels met de waarde 0 worden toegevoegd om de afbeelding heen. Door padding toe te voegen kan het filter over alle pixels van de initiële afbeelding heen (dus exclusief de toegevoegde nullen), en is er dus geen probleem bij de pixels in de hoeken. Hierdoor wordt het filter dus over de gehele afbeelding geplaatst uiteindelijk. Door middel van ReLu activatie worden deze nodes wel of niet geactiveerd. Vervolgens wordt het formaat van de afbeeldingen teruggebracht naar een kleiner formaat door MaxPooling per vier pixels toe te passen. Alleen de maximale waarde van zo’n cluster van vier pixels wordt dan meegenomen. Vervolgens gebeurt nogmaals hetzelfde met een Conv2D-laag van 64 filters en opnieuw pooling per vier pixels. 
Ten slotte worden de pixels door middel van Flatten weer bij elkaar gevoegd, en kan de foto geclassificeerd worden onder een van de klassen door middel van een Dense layer met een softmax activatiefunctie. Een softmax activatiefunctie wordt over het algemeen vaak gebruikt bij het classificeren van afbeeldingen met meerdere klassen. Een softmax functie berekent namelijk per klasse een getal dat aangeeft hoe groot de waarschijnlijkheid is dat de afbeelding tot die bepaalde klasse behoort. Het voordelige van het gebruiken van een softmax functie bij het classificeren met meerdere klassen is, is dat de grootste waarde van de softmax dichtbij 1 ligt, en alle andere waardes juist dichtbij 0. Hierdoor is het makkelijk te interpreteren tot welke klasse de afbeelding het meest waarschijnlijk bij hoort.  De som van alle outputs van de softmax functie heeft een waarde van 1, omdat alle waarschijnlijkheden bij elkaar opgeteld 100 procent is. Het aantal noden van de laatste laag is dus ook gelijk aan het aantal klassen vogelsoorten, 325, en per klasse is er dus een waarschijnlijkheid berekent.

Voor het trainen van het model wordt een aantal van 20 epochs aangehouden. Dit aantal is gekozen omdat dit voldoende mogelijkheden voor het model oplevert om te trainen, zonder onnodig veel verwerkingstijd te eisen zoals bij een hoger aantal epochs het geval is.


### Evaluatie en Conclusies
Evaluatie door analyseren van de training en validatie resultaten:
Na een aantal keer het model gerund te hebben is te zien dat het model overfit (zoals te zien is in de afbeelding hieronder). Dit is makkelijk te herkennen aan de trainingskosten die vrijwel op 0 zit na de 20 epochs terwijl de validatiekosten omhoog blijft gaan. Ook is te zien dat de accuraatheid van de validatiedata blijft steken tussen de 15 en 20% terwijl de trainingsdata een accuraatheid heeft van 95%. 

Figuur 1

Het huidige model dient als basismodel om vanuit verder te werken en is nog niet voldoende functioneel. Het classificeren van de verschillende vogelsoorten kan namelijk nog niet optimaal. De volgende stap is daarom om het model te optimaliseren.

Om ervoor te zorgen dat het model niet meer overfit kunnen een aantal verschillende mogelijkheden worden toegepast; batchnormalisatie, kruisvalidatie, dropout-methode en het vroegtijdig beëindigen van het trainen van het model. Batchnormalisatie houdt in dat de input van de lagen wordt genormaliseerd door deze opnieuw te centreren en opnieuw te schalen. Dit zorgt ervoor dat het netwerk stabieler en robuuster wordt door variërende input, hierdoor beter leert en daarbij gaat het ook deels overfitting tegen. Deze methode wordt echter over het algemeen pas toegepast wanneer het netwerk uit meerdere lagen bestaat en dus meer is verdiept. Ook kruisvalidatie kan leiden tot minder overfitten. Kruisvalidatie traint het model meerdere malen met k aantal verschillende gedeelten van de data. Het uiteindelijke model is dan getraind op kleinere inputs, maar is gemiddeld gezien wel door een representatie van de gehele data getraind. Daarnaast kan ook de dropout-methode tegen overfitting worden gebruikt. Hierbij wordt een gedeelte van de data niet meegenomen (‘gedropt’) naar een volgende layer die volgt op de dropout. Het netwerk overfit op deze manier minder omdat complexe aanpassingen aan de trainingsdata worden verminderd, maar het netwerk meer gegeneraliseerd functioneert. Als laatste kan er vroegtijdig worden gestopt met trainen wanneer de trainingskosten blijven dalen, maar de validatiekosten stijgen. Dit is namelijk het punt waar het model begint met overfitten. Wanneer het model op dit moment stopt met trainen kan het overfitten worden voorkomen. In het huidige model lijkt dit echter een voorbarige optie, waarbij de accuraatheid van het model alsnog op een lage waarde blijft steken. Verder kunnen er nog een aantal veranderingen worden doorgevoerd om te testen of het model hier accurater van wordt. Zo zouden er meer lagen aan het model toegevoegd kunnen worden. Door meer lagen toe te voegen, krijgt het model meer weights en wordt het model dus complexer. Dit zou ervoor kunnen zorgen dat het model dus meer features leert, en zo accurater wordt. Naast meer lagen toevoegen, kan er ook gebruik worden gemaakt van data augmentation. Data augmentation houdt in dat je meer data krijgt. Een voorbeeld van data augmentation is bijvoorbeeld horizontal flip, waarbij je elke afbeelding spiegelt op de horizontale as. Als je dit op elke afbeelding toepast, krijg je dus twee keer zo veel data. Andere voorbeelden van data augmentation zijn bijvoorbeeld rotatie en inzoomen. Data augmentation kan het model accurater maken omdat het ten eerste zorgt voor meer data, en het model kan hierdoor dus meer leren. Maar data augmentation zorgt er ook voor dat het model meer generaliseerbaar wordt naar nieuwe afbeeldingen. Stel je hebt een afbeelding van een papegaai van veraf. Door middel van data augmentation heb je de afbeelding ingezoomd, en hiermee ook het model getraind. Als er dan bij het testen van het model een afbeelding komt met een papegaai van dichtbij, dan is het waarschijnlijker dat het model dit correct geclassificeerd als er data augmentation was toegepast vergeleken met als dit niet was toegepast.

Zo kan er worden gekeken naar het eventueel toevoegen van andere activaties. Het model is nu getraind met ReLU activaties voor de hidden layers en een sigmoid activatie als laatst. Deze activatie kan worden aangepast naar verschillende activaties, bv.: softmax, tanh of ReLU. 


## Dropout-methode tegen overfitten

### Introductie
Uit de resultaten van het basismodel in het vorige hoofdstuk blijkt dat het model nog erg de neiging heeft tot overfitting. Al bij de eerste epochs wordt duidelijk dat de validatiekosten enorm stijgen en de trainingskosten heel laag worden, waaruit blijkt dat het model zeer nauwkeurig is afgestemd op de trainingsdata en niet goed bestand is tegen nieuwe datasets. De kern voor de aanpak van deze onnauwkeurigheid ligt in het selecteren van de data die het model gebruikt. Een methode om dit te verhelpen is de zogeheten dropout-methode. Bij de dropout-methode wordt een hidden layer toegevoegd aan het model, die ervoor zorgt dat met een bepaalde kans datapunten van de data worden verwijderd uit het model. Tijdens elke epoch wordt in deze laag deze kans voor elk datapunt toegepast, wat kan leiden tot het gebruiken van verschillende sets van datapunten bij het trainen van het model. De kracht van deze methode ligt in het feit dat bepaalde punten die vanwege grote uitschieters ervoor zorgen dat het model de trainingsdata erg specifiek nabootst, kunnen wegvallen en zo minder invloed hebben. Zo wordt het model robuuster voor invoer van nieuwe datasets, zoals de validatiedata.

#### Specifieke probleem
Het probleem dat getackeld dient te worden met deze ingreep is het overfitten van de trainingsdata door het model. Door op basis van kans datapunten in bepaalde epochs niet mee te nemen wordt de invloed van datapunten die veel ruis veroorzaken verkleind.

#### Veranderingen in het model
Aan het basismodel van hoofdstuk 1 zijn in een versie twee dropout-lagen toegevoegd, met een hoge kans voor datapunten om verwijderd te worden. Dit is een vrij extreme vorm van dropout, maar zo hopen wij de hoge mate van overfitting te bestrijden.
In een tweede versie is de kans van deze dropout-lagen gehalveerd, om zo het effect van de dropout te verminderen. 
Ten slotte is in een derde versie aan het basismodel slechts de eerste dropout-laag (tussen de twee Conv2D-lagen) met de gehalveerde kans toegevoegd. 

### Data Analyse en Voorverwerking
In het geval van de verbetering in dit hoofdstuk is slechts het toevoegen van hidden layers vereist, en dus geen voorverwerking van data.

### Model Pipeline en Training
De enige aanvulling op het basismodel uit hoofdstuk 2 zijn de dropout-lagen; de data die aan het model gevoed wordt en het aantal epochs blijft gelijk. 
In de eerste versie is allereerst tussen de twee Conv2D een dropout-laag met een kans van 0.4 toegevoegd, waarbij dus 40% van de datapunten wordt verwijderd en niet bijdraagt aan de verdere training van het model. Vervolgens is voor de flatten-laag een tweede dropout-laag met een kans van 0.2 toegevoegd.
De tweede versie is qua volgorde van het model volledig gelijk, alleen zijn de kansen gehalveerd naar 0.2 voor de eerste dropout-laag en 0.1 voor de tweede dropout-laag.
Ten slotte is in de derde versie slechts gebruik gemaakt van één dropout-laag, voor de flatten-laag met een kans van 0.2.

### Evaluatie en Conclusies
De resultaten van de eerste versie die dropout-lagen bevat zijn terug te vinden in figuur 2 en bevatten twee grafieken van kosten en precisie. Opmerkelijk is dat het overfitten van het basismodel tegengegaan lijkt te zijn, aangezien de trainingskosten en validatiekosten dichter bij elkaar liggen. Echter valt ook op dat de precisie van de validatiedata omlaag is gegaan ten opzichte van het basismodel. Dat terwijl het tegengaan van overfitting bedoeld is om het model beter werkend te maken op validatiedata zodat deze preciezer voorspeld kan worden. Met deze lagere precisie voor de validatiedata vormt deze eerste versie van dropout-layers geen waardevolle aanvulling op het basismodel. Dit kan eraan liggen dat de dropout een te grote invloed had op het model, en dat met het verwijderen van deze datapunten te veel informatie van het model verloren is gegaan. Het model underfit dan, wat ook blijkt uit dat de kosten voor de trainingsdata in dit model een stuk hoger zijn en dat de precisie van voorspelling nooit de 1 benaderd. Het is daarom verstandig om in versie 2 een dropout toe te passen met lagere kansen.

Figuur 2: Dropout 1: 0.4, dropout 2: 0.2

In figuur 3 is te zien dat deze mildere dropout inderdaad tot betere resultaten leidt; de precisie van zowel de trainingsdata als de validatiedata is gestegen, en de kosten zijn gedaald. Echter zijn deze waarden nog altijd minder wenselijk dan het basismodel, waardoor deze versie van dropout ook geen waardevolle aanvulling is op het basismodel. 


Figuur 3: Dropout 1: 0.2, dropout 2: 0.1

In figuur 4 met slechts de ene dropout-laag blijven de resultaten praktisch gelijk als voor de tweede versie. Hieruit valt te concluderen dat de aantasting van het basismodel vooral zit in de eerste dropout laag tussen de Conv2D-lagen in.


Figuur 4: Dropout 1: 0.2

Ondanks dat dropout in dit hoofdstuk op meerdere manieren is geprobeerd, is er geen versie gevonden die een verbetering van het basismodel oplevert. Wellicht dat bij een latere versie van het model dropout wel een verbetering van het model oplevert, maar in deze fase zullen wij teruggrijpen op het basismodel om op voort te bouwen. 

## Verdiepen van het netwerk

### Introductie
Een reden dat de dropout-methode niet werkt zoals gewenst is dat het netwerk nog niet complex genoeg is. Wanneer het netwerk niet complex genoeg is kan de dropout-methode de accuraatheid van het model verlagen, zoals in het model in het voorgaande hoofdstuk het geval was. Doordat bepaalde input wordt weggenomen, worden de nodes aan de hand van minder datapunten getraind en eindigen de weights op waardes die minder optimaal zijn. Hoewel het doel van dropout is dat deze weights inderdaad veranderen doordat er minder datapunten mee worden genomen en zo minder overfitten, blijft wel het doel dat de accuraatheid van het gehele model, en dus juist de accuraatheid op de validatiedata,  hierdoor toeneemt. Omdat dit doel niet werd bereikt is voor het huidige hoofdstuk besloten het netwerk te verdiepen en zo de accuraatheid van het model hopelijk te verhogen. Wanneer het model dan nog steeds overfit, kan er een nieuwe poging worden gedaan om het overfitten tegen te gaan. Dit zal dan in een nieuw hoofdstuk worden besproken.

#### Specifieke probleem
Het probleem van het huidige netwerk is dat deze nog niet complex genoeg is. De oplossing hiervoor is om het netwerk verder te verdiepen. Dit kan worden bereikt door extra lagen met nodes aan het netwerk toe te voegen.

#### Veranderingen in het model
Het huidige model neemt het basismodel uit hoofdstuk 2 opnieuw als basis. Aan dit model zijn twee extra Conv2D lagen toegevoegd inclusief MaxPooling lagen; een Conv2Dlaag met 128 nodes (en bijbehorende MaxPooling laag) en een Conv2D laag met 256 nodes (ook met bijbehorende MaxPooling laag).

### Data Analyse en Voorverwerking
Ook in het geval van de verbeteringen in dit hoofdstuk zijn geen aanpassingen van de data nodig. 

### Model Pipeline en Training
De data gebruikt voor de input is hetzelfde zoals in voorgaande hoofdstukken 2 en 3, evenals het aantal epochs (20). Als aanvulling op het basismodel is gekozen voor twee extra Conv2D lagen inclusief MaxPooling. De lagen bestaan ditmaal uit 128 en 256 nodes. Door het aantal nodes te vergroten kan het netwerk geïdentificeerde features tot op naar een steeds primairder basisniveau verkleinen. Opnieuw is ervoor gekozen om na de Conv2D lagen MaxPooling lagen toe te voegen die ervoor zorgen dat de extra tijd dat het trainen van het model kost met de twee extra lagen beperkt blijft.

### Evaluatie en Conclusies
De resultaten op de accuraatheid van het netwerk zijn terug te vinden in figuur 5.
De accuraatheid van de trainingsdata is nogmaals erg hoog, hoog in de 90%. De accuraatheid van de validatiedata daarentegen, is ditmaal toegenomen ten opzichte van het basismodel en ligt nu tussen de 50 en 55%. Dit is een positieve verandering, een toename in de accuraatheid van de validatiedata is gewenst om het netwerk te optimaliseren. Er is echter wel nog steeds sprake van overfitting, al is het verschil in de accuraatheid van de trainingsdata en validatie met het huidige verdiepte netwerk wel een stuk lager en overfit het model dus wel wat minder. Door het toevoegen van de extra lagen is het netwerk zeker complexer geworden en dit is terug te zien in de toegenomen accuraatheid van de validatiedata. Een volgende stap om het overfitten van het netwerk tegen te gaan is om nogmaals methodes die overfitten tegengaan uit te proberen op het nieuwe verdiepte netwerk.

Figuur 5: Kosten en accuraatheid van het verdiepte netwerk bestaande uit vier lagen.



## TEMPLATE

### Introductie
Wat gaan we doen, welk probleem pakken we aan met welke methode, uitleg methode. 
#### Specifieke probleem
… Specifieke probleem die deze versie probeert aan te pakken in 2 zinnen.
#### Veranderingen in het model
… High level overview van veranderingen van dit model t.o.v. het vorige model.

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
## TEMPLATE

### Introductie
Wat gaan we doen, welk probleem pakken we aan met welke methode, uitleg methode. 
#### Specifieke probleem
… Specifieke probleem die deze versie probeert aan te pakken in 2 zinnen.
#### Veranderingen in het model
… High level overview van veranderingen van dit model t.o.v. het vorige model.

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

