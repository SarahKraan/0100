# Eindrapport 
## Inhoudsopgave
[1: Inleiding](#1-inleiding)\
[2: Basismodel](#2-basismodel)\
[3: Drop-out methode tegen overfitten](#3-drop---out-methode-tegen-overfitten)\
[4: Verdiepen van het netwerk](#4-verdiepen-van-het-netwerk)\
[5: Leaky ReLu activatie functies](#5-leaky-relu-activatie-functies)\
[6: Normaliseren van de afbeeldingen](#6-normaliseren-van-de-afbeeldingen)\
[7: Data analyse](#7-data-analyse)\
[8: Data augmentatie (weighted)](#8-data-augmentatie-weighted)\
[9: Netwerk analyse](#9-netwerk-analyse)\
[10: Data augmentatie (horizontale flip)](#10-data-augmentatie-horizontale-flip)\
[11: Data augmentatie (croppen)](#11-data-augmentatie-croppen)\

## 1. Inleiding
Het aantal bedreigde diersoorten neemt elk jaar toe. Dit is te wijten aan onder andere de afnemende grootte van leefruimte voor dieren, het wegvallen van primaire of non-primaire voedselbronnen, een toenemend aantal stroperijen en klimaatveranderingen. Het is cruciaal om kennis te hebben van de diersoorten die in de nabije toekomst een groot risico lopen om een bedreigde diersoort te worden. Een hulpmiddel dat voor dit doeleinde kan worden gebruikt is het tellen van het aantal dieren dat van een bepaalde soort bestaat. Hierdoor kan in kaart worden gebracht of het aantal levende dieren van een bepaalde soort wellicht te laag is en er dus om actie wordt gevraagd om dit aantal omhoog te brengen door middel van bijvoorbeeld extra bescherming en het stimuleren van voortplanting. Echter, het proces van tellen is een enorm langdradig en routineus proces dat zich perfect zou lenen voor een meer geautomatiseerde vervanging in de vorm van een systeem dat gebruikt maakt van kunstmatige intelligentie (KI). Dit geautomatiseerde proces zou tot minder nodige mankracht én minder fouten kunnen leiden, waardoor er meer tijd en energie overblijft voor andere nuttige werkzaamheden.


## 2. Basismodel

### Introductie
In dit project wordt een convolutioneel neuraal netwerk ontwikkeld dat inspiratie haalt uit het menselijk brein. Het netwerk is opgebouwd uit meerdere lagen, en elke laag bevat een aantal nodes. De lagen zijn met elkaar verbonden, waarbij er gebruik wordt gemaakt van gewichten en biasen. Deze worden geleerd aan de hand van trainingsdata. 

Om het netwerk meer interpreteerbaar te maken, zou men de nodes kunnen beschrijven als neuronen, en de verbindingen tussen de nodes als de synapsen. Bij het menselijk brein zijn er twee type visuele neuronen: simpele en complexe neuronen. Simpele visuele neuronen identificeren de hoeken van lijnen in de afbeeldingen. Complexe visuele neuronen hebben een groter receptief veld dan simpele neuronen, en nemen hierbij de output van simpele neuronen om zo vervolgens de afbeeldingen te classificeren. Complexe visuele neuronen kijken zo dus naar de combinaties van de output gegeven door de simpele visuele neuronen. Ons convolutioneel netwerk bestaat uit twee verschillende fases, de ‘feature extraction’ fase en de classificatie fase. In de ‘feature extraction’ fase leert het model verschillende features van de vogels. Dit doet het netwerk aan de hand van convolution en pooling. Convolution komt overeen met de werking van simpele visuele neuronen. Het houdt namelijk in dat er een filter over de afbeelding wordt geplaatst om zo de plaatsing en oriëntatie van de randen te vinden. De output van een convolution wordt een feature map genoemd. Pooling komt overeen met de werking van complexe visuele neuronen, waarbij de hoeveelheid data wordt gereduceerd en de meest belangrijke data wordt behouden. In de classificatie fase worden vervolgens de feature maps bij elkaar genomen in een vector, en wordt het netwerk geleerd te voorspellen welke vogelsoort hoort bij de gegeven afbeelding van een vogel. Het doel van het netwerk in dit project is dus om te voorspellen tot welke soort de vogel op een afbeelding die aan het netwerk wordt meegegeven behoort. Er is hierbij een mogelijke uitkomst van één van 325 verschillende vogelsoorten. 

Het huidige project helpt mee aan het immense probleem omtrent het toenemende aantal van bedreigde diersoorten op meerdere manieren. Het netwerk dat tijdens dit project ontwikkeld wordt, biedt allereerst een manier om een afnemend aantal in bepaalde vogelsoorten vroegtijdig waar te nemen. Daarnaast doet het ontwikkelde netwerk dit op een manier waarbij minder beroep op de mens wordt gedaan en waarbij het tellen van vogels zowel sneller als met minder fouten kan plaatsvinden. Al met al belooft het huidige project een grote bijdrage aan het beschermen van een groot aantal vogelsoorten tegen uitsterven. 

#### Specifieke probleem
Het specifieke probleem dat in dit gedeelte van het eindrapport wordt behandeld is het in kaart brengen van de beschikbare data en haar eigenschappen. Verder is er een eerste model van het netwerk ontwikkeld dat als basis dient voor de komende versies en aanpassingen van het netwerk in de hoofdstukken die volgen.

#### Overzicht model
Voor de opbouw van het basismodel is gebruik gemaakt van de TensorFlow bibliotheek. Om te beginnen zijn een aantal basislagen aan het model toegevoegd. Hierbij zijn om en om twee Conv2D lagen en MaxPooling lagen gebruikt voor de ‘feature extraction’ fase, gevolgd door een Flatten en Dense laag voor de classificatie fase. Er is gekozen om te beginnen met twee Conv2D en Maxpooling lagen in de ‘feature extraction’ fase, om zo het model de features te leren van de verschillende vogelsoorten. Omdat dit nog het basismodel is, hebben we besloten het model simpel te houden, en dus nog niet al te veel lagen toe te voegen. Zo kan namelijk gecheckt worden hoe opvallend de features van de verschillende vogelsoorten zijn, en hoe goed ze dus op een basisniveau al te onderscheiden zijn van elkaar. Verder hebben we in de classificatie fase één flatten laag, omdat de verschillende feature maps maar een keer tot een vector genomen hoeven te worden. Het toevoegen van slechts een dense layer is weer om het model zo simpel mogelijk te houden, en te kijken hoe goed het model al voorspelt als het nog niet al te complex is gemaakt.

![image](https://user-images.githubusercontent.com/68432564/150953829-d48db968-be4f-4bde-87c1-4b0fba2d10be.png)

_Figuur 1:_ Model opbouw 

### Data Analyse en Voorverwerking
De dataset 325 Birds Species te vinden op [kaggle](https://www.kaggle.com/gpiosenka/100-bird-species) voorziet ons van alle data die nodig is om dit onderzoek uit te voeren. Deze dataset bestaat uit 50582 foto’s in totaal van 325 soorten vogels.  Elke soort vogel betreft minstens 120 afbeeldingen, zowel de mannelijke als de vrouwelijke variant. De afbeeldingen hebben als afmeting allemaal 224 (pixels) x 224 (pixels) x 3 (kanalen). De drie kanalen houden in dat de afbeeldingen volgens het RGB-systeem gekleurd zijn. Hieronder volgen een aantal voorbeeldafbeeldingen uit de dataset: 

![image](https://user-images.githubusercontent.com/59557088/149712632-efcb392b-8414-4fa2-88e3-2b93c935677d.png)![image](https://user-images.githubusercontent.com/59557088/149712645-67429430-3e64-4bc8-9b5b-95367ac61687.png)![image](https://user-images.githubusercontent.com/59557088/149712619-b6f44347-bdc7-4648-be41-a59847b5dd03.png)

_Afbeelding 1:_ Een aantal afbeeldingen uit de dataset ter illustratie.

Binnen de dataset was al onderscheid gemaakt tussen training-, test- en validatiedata; de trainingsdata bevat 47332 afbeeldingen en de test- en validatiedata allebei 1625 afbeeldingen. Deze verdeling is zo optimaal mogelijk gedaan om een zo precies mogelijke voorspelling over de vogelsoort bij een afbeelding te kunnen maken. Vanwege deze verdeling die al in de dataset aanwezig was, is het niet nodig om een eigen onderverdeling te maken. 

Als eerste werd alle data ingeladen. Om RAM problemen te voorkomen zijn de formaten van de afbeeldingen verkleind naar 64 x 64 pixels. We hebben dit gedaan omdat het programma dan ongeveer drie keer zo snel werkt en dus minder snel crasht. Omdat 64 een getal is wat voortkomt uit een 2 macht kan hier makkelijk mee gerekend worden (26 = 64).

Ook zijn de labels van de afbeeldingen omgezet in een matrixvorm door zogeheten ‘one-hot encoding’, waarbij een 1 wordt geplaatst bij het label van de juiste categorie vogel en bij de andere categorieën een 0. Op deze manier kan de matrix van labels vergeleken worden met de uitkomst van het model.


### Model Pipeline en Training
Het basismodel neemt als input de trainingsdata die bestaat uit 47332 afbeeldingen verwerkt in een matrix van afmeting 47332 x 85 x 85  x 3 en de bijbehorende matrix van labels met afmeting 47332 x 325. Deze afbeeldingen worden gevoed aan de eerste laag van het model, een Conv2D-laag van 32 filters. Per afbeelding wordt padding met specificatie ‘same’ toegevoegd, waarbij er pixels met pixelwaarde 0 worden toegevoegd om de afbeelding heen. Door padding toe te voegen kan het filter over alle pixels van de initiële afbeelding heen (dus exclusief de toegevoegde nullen), en is er dus geen probleem bij de pixels in de hoeken. Hierdoor wordt het filter dus over de gehele afbeelding geplaatst uiteindelijk. Door middel van ReLu activatie worden deze nodes wel of niet geactiveerd. Vervolgens wordt het formaat van de afbeeldingen teruggebracht naar een kleiner formaat door MaxPooling per vier pixels toe te passen. Alleen de maximale waarde van zo’n cluster van vier pixels wordt dan meegenomen. Vervolgens gebeurt nogmaals hetzelfde met een Conv2D-laag van 64 filters en opnieuw pooling per vier pixels. 

Ten slotte worden van de pixels door middel van Flatten een vector gemaakt, en kan de foto geclassificeerd worden onder een van de klassen door middel van een Dense layer met een softmax activatiefunctie. Een softmax activatiefunctie wordt over het algemeen vaak gebruikt bij het classificeren van afbeeldingen met meerdere klassen (1). Een softmax functie berekent namelijk eerst per afbeelding voor elke klasse een getal dat aangeeft in hoeverre de afbeelding tot die bepaalde klasse behoort. Hierna worden deze waardes genormaliseerd naar waarschijnlijkheden volgens een waarschijnlijkheidsverdeling. Het aantal noden van de laatste laag is dan gelijk aan het aantal klassen vogelsoorten, 325, en per klasse is er dus een waarschijnlijkheid berekend. Het voordelige van het gebruiken van een softmax functie bij het classificeren met meerdere klassen is dat de softmax dan dus een waarde voorspelt tussen 0 en 1 voor elke klasse, waarbij de waardes bij elkaar opgeteld 1 zijn en hierdoor voor elke klasse de waarschijnlijkheid weergeeft dat een afbeelding tot deze klasse behoort. Dit zorgt voor een intuïtieve uitkomst die gemakkelijk interpreteerbaar is. 

Voor het trainen van het model wordt een aantal van 20 epochs aangehouden. Dit aantal is gekozen omdat dit voldoende mogelijkheden voor het model oplevert om te trainen, zonder onnodig veel verwerkingstijd te eisen zoals bij een hoger aantal epochs het geval is. Om te kijken hoe goed het model het doet kan worden gekeken naar de kosten die het model maakt en de accuraatheid van het model. Als we dit in 2 aparte plots zetten en de training en de validatie data tegen elkaar uitzetten kan zo een goed beeld geschetst worden van de vooruitgang. De kosten worden berekend d.m.v. het toepassen van de categorical crossentropy functie. Deze functie is goed voor het classificeren omdat de voorspellingen altijd tussen de 0 en 1 uit zullen komen; 0 voor zeker niet en 1 voor zeker wel. 

### Evaluatie en Conclusies
Evaluatie door analyseren van de training en validatie resultaten:
Na een aantal keer het model gerund te hebben is te zien dat het model overfit (zoals te zien is in de afbeelding hieronder). Dit is makkelijk te herkennen aan de trainingskosten die vrijwel op 0 zit na de 20 epochs terwijl de validatiekosten omhoog blijft gaan. Ook is te zien dat de accuraatheid van de validatiedata blijft steken tussen de 15 en 20% terwijl de trainingsdata een accuraatheid heeft van 95%. 

![image](https://user-images.githubusercontent.com/68432564/150954540-90615182-1eb1-4137-a656-4e3a4829d797.png)

_Figuur 2:_ Model resultaten basismodel

Het huidige model dient als basismodel om vanuit verder te werken maar is nog niet op het gewenste niveau. Het classificeren van de verschillende vogelsoorten gaat namelijk nog niet optimaal. De volgende stap is daarom om het model te optimaliseren.

Om ervoor te zorgen dat het model niet meer overfit kunnen een aantal verschillende mogelijkheden worden toegepast; batchnormalisatie, kruisvalidatie, dropout-methode, en het vroegtijdig beëindigen van het trainen van het model. 

Batchnormalisatie houdt in dat de input van de lagen wordt genormaliseerd door deze opnieuw te centreren en opnieuw te schalen. Dit wordt vaak gedaan aan de hand van aanleerbare parameters (gamma en beta). De parameters worden geüpdate door middel van gradient descent. Batchnormalisatie helpt bij overfitting, omdat het de covariate shift reduceert. Covariate shift houdt in dat de distributie van de input verschilt tussen verschillende datasets. Dit kan bijvoorbeeld voorkomen als de data wordt getraind met zwarte katten, maar de testdata vervolgens voornamelijk bestaat uit gekleurde katten. Om dezelfde distributie van data te behouden, worden dus de waardes van de input steeds genormaliseerd, en wordt het model zo dus ook accurater. Ook leert elke laag zo een beetje zelfstandig de weights. Naast dat batchnormalisatie helpt bij covariate shift, regulariseert het ook lichtelijk. Dit omdat door batchnormalisatie elke laag een zekere mate van noise krijgt. Door noise toe te voegen wordt het netwerk meer robuust, en is het zo minder ingesteld op de training data. Het model wordt dus simpeler, en hierdoor overfit het minder snel.

Ook kruisvalidatie kan leiden tot minder overfitten. Kruisvalidatie traint het model meerdere malen met k aantal verschillende gedeelten van de data. Het uiteindelijke model is dan getraind op kleinere inputs, maar is gemiddeld gezien wel door een representatie van de gehele data getraind. 

Daarnaast kan ook de dropout-methode tegen overfitting worden gebruikt. Hierbij wordt een gedeelte van de data gelijkgesteld aan 0 (‘gedropt’) waardoor deze niet bijdraagt aan het trainen van het model. Het netwerk overfit op deze manier minder omdat complexe aanpassingen aan de trainingsdata worden verminderd, maar het netwerk meer gegeneraliseerd functioneert. 

Als laatste kan er vroegtijdig worden gestopt met trainen wanneer de trainingskosten blijven dalen, maar de validatiekosten stijgen. Dit is namelijk het punt waar het model begint met overfitten. Wanneer het model op dit moment stopt met trainen kan het overfitten worden voorkomen. In het huidige model lijkt dit echter een voorbarige optie, waarbij de accuraatheid van het model alsnog op een lage waarde blijft steken. 

Verder kunnen er nog een aantal veranderingen worden doorgevoerd om te testen of het model hier accurater van wordt. 

Zo zouden er meer lagen aan het model toegevoegd kunnen worden. Door meer lagen toe te voegen, krijgt het model meer weights en wordt het model dus complexer. Dit zorgt ervoor dat het model meer complexere features leert, en zo accurater wordt. 

Naast het toevoegen van meer lagen, kan er ook gebruik worden gemaakt van data augmentatie. Data augmentatie houdt in dat je meer data krijgt. Een voorbeeld van data augmentatie is bijvoorbeeld horizontal flip, waarbij je elke afbeelding spiegelt op de horizontale as. Als je dit op elke afbeelding toepast, krijg je dus twee keer zo veel data. Andere voorbeelden van data augmentatie zijn bijvoorbeeld rotatie en inzoomen. Data augmentatie kan het model accurater maken omdat het ten eerste zorgt voor meer data, en het model kan hierdoor dus meer leren. Maar data augmentatie zorgt er ook voor dat het model meer generaliseerbaar wordt naar nieuwe afbeeldingen. Stel je hebt een afbeelding van een papegaai van veraf. Door middel van data augmentatie heb je de afbeelding ingezoomd, en hiermee ook het model getraind. Als er dan bij het testen van het model een afbeelding komt met een papegaai van dichtbij, dan is het waarschijnlijker dat het model dit correct geclassificeerd als er data augmentatie was toegepast vergeleken met als dit niet was toegepast. Data augmentatie kan op de gehele trainingsdata worden toegepast of een deel hiervan. Een eerste stap zou dan zijn het onderzoeken van op welke data de data augmentatie wel toe te passen en op welke data niet.

Als eerste zal worden geprobeerd het overfitten van het basismodel tegen te gaan.

## 3. Dropout-methode tegen overfitten

### Introductie
Uit de resultaten van het basismodel in het vorige hoofdstuk blijkt dat het model nog erg de neiging heeft tot overfitting. Al bij de eerste epochs wordt duidelijk dat de validatiekosten enorm stijgen en de trainingskosten heel laag worden, waaruit blijkt dat het model zeer nauwkeurig is afgestemd op de trainingsdata en niet goed bestand is tegen nieuwe datasets. De kern voor de aanpak van deze onnauwkeurigheid ligt in het selecteren van de data die het model gebruikt. 

Een methode om dit te verhelpen is de zogeheten dropout-methode. Bij de dropout-methode wordt een hidden layer toegevoegd aan het model, die ervoor zorgt dat met een bepaalde kans delen van de input van de laag naar een waarde van 0 worden gezet. Hierdoor worden zo dus bepaalde nodes in deze laag genegeerd tijdens het trainen van het model. Tijdens elke epoch wordt in deze laag deze kans voor elk input toegepast, wat kan leiden tot het gebruiken van verschillende sets van nodes bij het trainen van het model. De kracht van deze methode ligt in het feit dat bepaalde punten die vanwege grote uitschieters ervoor zorgen dat het model de trainingsdata erg specifiek nabootst, kunnen wegvallen en zo minder invloed hebben. Zo wordt het model robuuster voor invoer van nieuwe datasets, zoals de validatiedata.

#### Specifieke probleem
Het probleem dat getackeld dient te worden met deze ingreep is het overfitten van de trainingsdata door het model. Door op basis van kans delen van de input in bepaalde epochs niet mee te nemen wordt de invloed van input die veel ruis veroorzaakt verkleind.

#### Veranderingen in het model
Aan het basismodel van hoofdstuk 1 zijn in een versie twee dropout-lagen toegevoegd, met een hoge kans voor nodes om naar een waarde van 0 gezet te worden. Dit is een vrij extreme vorm van dropout, maar zo hopen wij de hoge mate van overfitting te bestrijden.

In een tweede versie is de kans van deze dropout-lagen gehalveerd, om zo het effect van de dropout te verminderen. 

Ten slotte is in een derde versie aan het basismodel slechts de eerste dropout-laag (tussen de twee Conv2D-lagen) met de gehalveerde kans toegevoegd. 

![image](https://user-images.githubusercontent.com/68432564/150954711-0938a785-cb09-44c5-9175-e7388b451a8d.png)

_Figuur 3:_ Model opbouw 

### Evaluatie en Conclusies
De resultaten van de eerste versie die dropout-lagen bevat zijn terug te vinden in figuur 2 en bevatten twee grafieken van kosten en precisie. Opmerkelijk is dat het overfitten van het basismodel tegengegaan lijkt te zijn, aangezien de trainingskosten en validatiekosten dichter bij elkaar liggen. Echter valt ook op dat de precisie van de validatiedata omlaag is gegaan ten opzichte van het basismodel. Dat terwijl het tegengaan van overfitting bedoeld is om het model beter werkend te maken op validatiedata zodat deze preciezer voorspeld kan worden. Met deze lagere precisie voor de validatiedata vormt deze eerste versie van dropout-layers geen waardevolle aanvulling op het basismodel. Dit kan eraan liggen dat de dropout een te grote invloed had op het model, en dat met het verwijderen van deze datapunten te veel informatie van het model verloren is gegaan. Het model underfit dan, wat ook blijkt uit dat de kosten voor de trainingsdata in dit model een stuk hoger zijn en dat de precisie van voorspelling nooit de 1 benaderd. Het is daarom verstandig om in versie 2 een dropout toe te passen met lagere kansen.

 ![image](https://user-images.githubusercontent.com/59557088/149712139-7a261f6f-d1ba-4b65-8862-cc78de30fc33.png)

Figuur 2: Dropout 1: 0.4, dropout 2: 0.2

In figuur 3 is te zien dat deze mildere dropout inderdaad tot betere resultaten leidt; de precisie van zowel de trainingsdata als de validatiedata is gestegen, en de kosten zijn gedaald. Echter zijn deze waarden nog altijd minder wenselijk dan het basismodel, waardoor deze versie van dropout ook geen waardevolle aanvulling is op het basismodel. 

 ![image](https://user-images.githubusercontent.com/59557088/149712165-3e90767c-3226-4058-a8d2-18bc81d5d23f.png)

Figuur 3: Dropout 1: 0.2, dropout 2: 0.1

In figuur 4 met slechts de ene dropout-laag blijven de resultaten praktisch gelijk als voor de tweede versie. Hieruit valt te concluderen dat de aantasting van het basismodel vooral zit in de eerste dropout laag tussen de Conv2D-lagen in.

 ![image](https://user-images.githubusercontent.com/59557088/149712194-817606b3-2abd-4c3e-920f-5073e924c2f1.png)

Figuur 4: Dropout 1: 0.2

Ondanks dat dropout in dit hoofdstuk op meerdere manieren is geprobeerd, is er geen versie gevonden die een verbetering van het basismodel oplevert. Wellicht dat bij een latere versie van het model dropout wel een verbetering van het model oplevert, maar in deze fase zullen wij teruggrijpen op het basismodel om op voort te bouwen. 

## Verdiepen van het netwerk

### Introductie
Een reden dat de dropout-methode niet werkt zoals gewenst is dat het netwerk nog niet complex genoeg is. Wanneer het netwerk niet complex genoeg is kan de dropout-methode de accuraatheid van het model verlagen, zoals in het model in het voorgaande hoofdstuk het geval was. Doordat bepaalde input wordt weggenomen, worden de nodes aan de hand van minder datapunten getraind en eindigen de weights op waardes die minder optimaal zijn. Hoewel het doel van dropout is dat deze weights inderdaad veranderen doordat er minder datapunten mee worden genomen en zo minder overfitten, blijft wel het doel dat de accuraatheid van het gehele model, en dus juist de accuraatheid op de validatiedata,  hierdoor toeneemt. Omdat dit doel niet werd bereikt is voor het huidige hoofdstuk besloten het netwerk te verdiepen om zo de accuraatheid van het model hopelijk te verhogen. Wanneer het model dan nog steeds overfit, kan er een nieuwe poging worden gedaan om het overfitten tegen te gaan. Dit zal dan in een nieuw hoofdstuk worden besproken.

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
De accuraatheid van de trainingsdata is nogmaals erg hoog, hoog in de 90%. De accuraatheid van de validatie data daarentegen, is ditmaal toegenomen ten opzichte van het basismodel en ligt nu tussen de 50 en 55%. Dit is een positieve verandering, een toename in de accuraatheid van de validatie data is gewenst om het netwerk te optimaliseren. Er is echter wel nog steeds sprake van overfitting, al is het verschil in de accuraatheid van de trainings- en validatie data met het huidige verdiepte netwerk wel een stuk lager en overfit het model dus wel wat minder. Door het toevoegen van de extra lagen is het netwerk zeker complexer geworden en dit is terug te zien in de toegenomen accuraatheid van de validatie data.

 ![image](https://user-images.githubusercontent.com/59557088/149712239-bcc3ab2c-4924-49da-9922-c1097eea15a1.png)

Figuur 5: Kosten en accuraatheid van het verdiepte netwerk bestaande uit vier lagen.

## Leaky ReLu activatie functies

### Introductie
Om overfitten tegen te gaan is een veelbelovende methode het aanpassen van de activatie functies van de verborgen lagen. Het huidige model bevat vier Conv2D-lagen waarin de nodes elk geactiveerd worden met ReLu-activaties. Deze activatiecode levert in de helft van de gevallen een activatiewaarde van 0 op, waarmee deze nodes niet meewerken bij het trainen van het model en ‘dode nodes’ zijn. Bij zeer weinig actieve nodes wordt de trainingsdata zeer precies nagebootsts en overfit het model gemakkelijk. Door de activatie functies aan te passen naar Leaky ReLu-activaties zullen er niet zoveel activatie waardes van 0 meer zijn, maar zullen deze activatie waardes bestaan uit negatieve waardes (onder de 0). Omdat de nodes niet meer verloren gaan, leveren de nodes nog steeds een bijdrage aan het trainen van het model. Hierdoor wordt het probleem met vele ‘dode nodes’ verholpen, en werken deze nodes wel mee om zo hopelijk overfitting tegen te gaan. 

#### Specifieke probleem
Het huidige probleem is nog steeds het overfitten van het huidige model. De oplossing die in het huidige hoofdstuk wordt aangedragen zijn het wijzigen van de ReLu-activaties naar Leaky ReLu-activaties. De nodes die door ReLu-activaties dode nodes opleveren worden met Leaky ReLu-activaties wel actief, en helpen zo het model minder specifiek de trainingsdata fitten om zo minder te overfitten.

#### Veranderingen in het model
In elke van de vier Conv2D-lagen is de ReLu-activatie functie vervangen door een Leaky-ReLu functie. De alpha - de hoek van de Leaky-ReLu functie- is in verschillende versies gevarieerd om zo de optimale waarde te vinden. 

### Data Analyse en Voorverwerking
Bij deze verbeterstap is geen data analyse of voorverwerking van pas gekomen maar is de data gebruikt zoals in de vorige hoofdstukken.

### Model Pipeline en Training
In versie 1 van deze verbeterstap is de alpha van de Leaky-ReLu functies ingesteld op 0.1.
In versie 2 van deze verbeterstap is de alpha van de Leaky-ReLu functies verhoogd naar 0.2 om te kijken of dit een grotere validatie precisie oplevert. 
In versie 3 van deze verbeterstap is de alpha van de Leaky-ReLu functies nogmaals verhoogd naar 0.3.

### Evaluatie en Conclusies

In figuur 6 zijn de resultaten te zien van de eerste versie met een alpha van 0.1. Opmerkelijk is dat de validatie precisie een flinke sprong heeft gemaakt van 0.52 naar 0.60. Wat dat betreft heeft deze nieuwe activatiefunctie een zeer positieve invloed gehad. Echter is de vorm van de grafieken niet zozeer aangepast en blijft de precisie van de trainingsdata zowat gelijk, dus blijft het probleem van overfitting bestaan. Ook zijn de kosten voor validatie van het model gestegen.


![image](https://user-images.githubusercontent.com/68432564/149932671-09b721f8-5663-4ea2-94f8-d62884a653f0.png)

Figuur 6: Kosten en accuraatheid met een alpha van 0.1 voor de Leaky-ReLu activaties

In figuur 7 zijn de resultaten weergegeven wanneer de alpha is verhoogd naar 0.2. Wederom is de validatie precisie toegenomen, hoewel marginaal tot 0.62. Ook zijn de validatie kosten nog verder toegenomen.

![image](https://user-images.githubusercontent.com/68432564/149932730-b3366f9c-05e5-48f5-9a7b-96816b1dfe52.png)

Figuur 7: Kosten en accuraatheid met een alpha van 0.2 voor de Leaky-ReLu activaties

In figuur 8 valt te zien dat de hogere alpha van 0.3 geen grotere validatie precisie meer oplevert; deze is gedaald tot 0.58. Daarentegen zijn de validatie kosten wel weer verder toegenomen. Blijkbaar is de Leaky-ReLu functie hier dus te steil aflopend.

![image](https://user-images.githubusercontent.com/68432564/149932768-afd0de69-df9a-42d0-9965-ab72fd9160ff.png)

Figuur 8: Kosten en accuraatheid met een alpha van 0.3 voor de Leaky-ReLu activaties

Na het testen van deze verschillende versies kan geconcludeerd worden dat een Leaky-ReLu activatie met een alpha van 0.2 de beste aanvulling vormt op ons basismodel. De validatie precisie is daarmee hoger en dat is waar naar gestreefd dient te worden. Om de aanhoudende overfitting van het netwerk verder tegen te gaan lijkt het ons verstandig om in de volgende vervolgstap te onderzoeken bij welke data het netwerk het meest de fout in gaat. Een goede vervolgstap zou daarom zijn om te analyseren bij welke vogelsoorten de classificatie niet correct verloopt, en de data hierop te augmenteren. Het lijkt ons verstandig om zo eerst de prestatie van het model bij de input aan te pakken, om vervolgens pas weer naar verdere verbeteringen tegen overfitting te kijken.


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

Bronnen:
1). https://towardsdatascience.com/analyzing-different-types-of-activation-functions-in-neural-networks-which-one-to-prefer-e11649256209
2).
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator


