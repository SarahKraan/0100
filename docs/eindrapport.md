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

Als eerste werd alle data ingeladen. Om RAM problemen te voorkomen zijn de formaten van de afbeeldingen verkleind naar 64 x 64 pixels. We hebben dit gedaan omdat het programma dan ongeveer drie keer zo snel werkt en dus minder snel crasht. Omdat 64 een getal is wat voortkomt uit een 2 macht kan hier makkelijk mee gerekend worden ![formula](https://render.githubusercontent.com/render/math?math=2^{6} = 64)

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

### Data Analyse en Voorverwerking
In het geval van de verbetering in dit hoofdstuk is slechts het toevoegen van hidden layers vereist, en dus geen voorverwerking van data.

### Model Pipeline en Training
De enige aanvulling op het basismodel uit hoofdstuk 2 zijn de dropout-lagen; de data die aan het model gevoed wordt en het aantal epochs blijft gelijk. 

In de eerste versie is allereerst tussen de twee Conv2D een dropout-laag met een kans van 0.4 toegevoegd, waarbij dus 40% van de nodes in deze laag naar 0 wordt gezet. Vervolgens is voor de flatten-laag een tweede dropout-laag met een kans van 0.2 toegevoegd.

De tweede versie is qua volgorde van het model volledig gelijk, alleen zijn de kansen gehalveerd naar 0.2 voor de eerste dropout-laag en 0.1 voor de tweede dropout-laag.

Ten slotte is in de derde versie slechts gebruik gemaakt van één dropout-laag. Deze is geplaatst tussen de twee Conv2D lagen in.

### Evaluatie en Conclusies
De resultaten van de eerste versie die dropout-lagen bevat zijn terug te vinden in figuur 2 en bevatten twee grafieken van kosten en accuraatheid. Opmerkelijk is dat het overfitten van het basismodel tegengegaan lijkt te zijn, aangezien de trainingskosten en validatiekosten dichter bij elkaar liggen. Echter valt ook op dat de accuraatheid van de validatiedata omlaag is gegaan ten opzichte van het basismodel. Dat terwijl het tegengaan van overfitting bedoeld is om het model beter werkend te maken op validatiedata zodat deze preciezer voorspeld kan worden. Met deze lagere accuraatheid voor de validatiedata vormt deze eerste versie van dropout-layers geen waardevolle aanvulling op het basismodel. Dit kan eraan liggen dat de dropout een te grote invloed had op het model, en dat met het naar 0 zetten van deze nodes te veel informatie van het model verloren is gegaan. Het model underfit dan, wat ook blijkt uit dat de kosten voor de trainingsdata in dit model een stuk hoger zijn en dat de accuraatheid van voorspelling nooit de 1 benaderd. Het is daarom verstandig om in versie 2 een dropout toe te passen met lagere kansen.

![image](https://user-images.githubusercontent.com/68432564/150955062-2a39cdd8-9fbd-4876-b5cd-a5d066f76a3f.png)

_Figuur 4:_ Dropout 1: 0.4, dropout 2: 0.2

In figuur 5 is te zien dat deze mildere dropout inderdaad tot betere resultaten leidt; de accuraatheid van zowel de trainingsdata als de validatiedata is gestegen, en de kosten zijn gedaald. Echter zijn deze waarden nog altijd minder wenselijk dan het basismodel, waardoor deze versie van dropout ook geen waardevolle aanvulling is op het basismodel. 

![image](https://user-images.githubusercontent.com/68432564/150955122-aaec9143-1f67-4361-aaa5-c810ae47aefd.png)

_Figuur 5:_ Dropout 1: 0.2, dropout 2: 0.1

In figuur 6 met slechts de ene dropout-laag blijven de resultaten praktisch gelijk als voor de tweede versie. Hieruit valt te concluderen dat de aantasting van het basismodel vooral zit in de eerste dropout laag tussen de Conv2D-lagen in.

![image](https://user-images.githubusercontent.com/68432564/150955191-a1aaa68b-ec20-47b4-88a3-00a7680a0fa4.png)

_Figuur 6:_ Dropout 1: 0.2

Ondanks dat dropout in dit hoofdstuk op meerdere manieren is geprobeerd, is er geen versie gevonden die een verbetering van het basismodel oplevert. Weliswaar is de mate van overfitting afgenomen doordat de trainings- en validatie resultaten dichterbij elkaar komen te liggen, maar dit gaat dusdanig ten koste van de validatie accuraatheid dat dit geen verbetering op het model oplevert. Wellicht dat bij een latere versie van het model dropout wel een verbetering van het model oplevert, maar in deze fase zullen wij teruggrijpen op het basismodel om op voort te bouwen. 

## 4. Verdiepen van het netwerk

### Introductie
Een mogelijke reden voor het niet werken van de dropout-methode kan zijn dat het netwerk nog niet complex genoeg is. Wanneer het netwerk niet complex genoeg is kan de dropout-methode de accuraatheid van het model verlagen, zoals in het model in het voorgaande hoofdstuk het geval was. Doordat bepaalde input wordt weggenomen, worden de nodes aan de hand van minder datapunten getraind en eindigen de weights op waardes die minder optimaal zijn. Hoewel het doel van dropout is dat deze weights inderdaad veranderen doordat er minder input mee wordt genomen en zo minder overfitten, blijft wel het doel dat de accuraatheid van het gehele model, en dus juist de accuraatheid op de validatiedata, hierdoor toeneemt. 

Omdat dit doel niet werd bereikt is voor het huidige hoofdstuk besloten het netwerk te verdiepen om zo de accuraatheid van het model hopelijk te verhogen. Wanneer het model dan nog steeds overfit, kan er een nieuwe poging worden gedaan om het overfitten tegen te gaan. Dit zal dan in een nieuw hoofdstuk worden besproken.

#### Specifieke probleem
Het probleem van het beginmodel is dat dit nog niet complex genoeg is. De oplossing hiervoor is om het netwerk verder te verdiepen. Dit kan worden bereikt door extra lagen met nodes aan het netwerk toe te voegen.

#### Veranderingen in het model
Het huidige model neemt het basismodel uit hoofdstuk 2 opnieuw als basis. Aan dit model zijn twee extra Conv2D lagen toegevoegd inclusief MaxPooling lagen; een Conv2Dlaag met 128 nodes (en bijbehorende MaxPooling laag) en een Conv2D laag met 256 nodes (ook met bijbehorende MaxPooling laag).

![image](https://user-images.githubusercontent.com/68432564/150955332-0cb7b2fc-d119-4102-bab7-287e5f6edea2.png)

_Figuur 7:_ Model opbouw

### Data Analyse en Voorverwerking
Ook in het geval van de verbeteringen in dit hoofdstuk zijn geen aanpassingen van de data nodig. 

### Model Pipeline en Training
De data gebruikt voor de input is hetzelfde zoals in voorgaande hoofdstukken 2 en 3, evenals het aantal epochs (20). 

Als aanvulling op het basismodel is gekozen voor twee extra Conv2D lagen inclusief MaxPooling. De lagen bestaan ditmaal uit 128 en 256 nodes. Door het aantal nodes te vergroten kan het netwerk geïdentificeerde features tot op naar een steeds primairder basisniveau verkleinen. Opnieuw is ervoor gekozen om na de Conv2D lagen MaxPooling lagen toe te voegen die ervoor zorgen dat de extra tijd dat het trainen van het model kost met de twee extra lagen beperkt blijft.

### Evaluatie en Conclusies
De resultaten op de accuraatheid van het netwerk zijn terug te vinden in figuur 8.
De accuraatheid van de trainingsdata is nogmaals erg hoog, hoog in de 90%. De accuraatheid van de validatie data daarentegen, is ditmaal toegenomen ten opzichte van het basismodel en ligt nu tussen de 50 en 55%. Dit is een positieve verandering, een toename in de accuraatheid van de validatie data is gewenst om het netwerk te optimaliseren. Er is echter wel nog steeds sprake van overfitting, al is het verschil in de accuraatheid van de trainings- en validatie data met het huidige verdiepte netwerk wel een stuk lager en overfit het model dus wel wat minder. Door het toevoegen van de extra lagen is het netwerk zeker complexer geworden en dit is terug te zien in de toegenomen accuraatheid van de validatie data.

![image](https://user-images.githubusercontent.com/68432564/150955406-e3cc9042-4eab-4624-9d2c-1717e92e44fb.png)

_Figuur 8:_ Kosten en accuraatheid van het verdiepte netwerk bestaande uit vier lagen.

Een volgende stap om het overfitten van het netwerk tegen te gaan is om nogmaals methodes die overfitten tegengaan uit te proberen op het nieuwe verdiepte netwerk. Een veelbelovende methode om het overfitten tegen te gaan zou het aanpassen van de activatie functies kunnen zijn bij de verborgen lagen. Tot nu toe is gebruik gemaakt van de ReLu activatie functie. Een activatie functie die bij het huidige model minder neiging tot overfitten zou kunnen hebben is de Leaky ReLu functie.

## 5. Leaky ReLu activatie functies

### Introductie
Om overfitten tegen te gaan is een veelbelovende methode het aanpassen van de activatie functies van de verborgen lagen. Het huidige model bevat vier Conv2D-lagen waarin de nodes elk geactiveerd worden met ReLu-activaties. Deze activatiecode levert in de helft van de gevallen een activatiewaarde van 0 op, waarmee deze nodes niet meewerken bij het trainen van het model en ‘dode nodes’ zijn. Bij zeer weinig actieve nodes wordt de trainingsdata zeer precies nagebootsts en overfit het model gemakkelijk. Door de activatie functies aan te passen naar Leaky ReLu-activaties zullen er niet zoveel activatie waardes van 0 meer zijn, maar zullen deze activatie waardes bestaan uit negatieve waardes (onder de 0). Omdat de nodes niet meer verloren gaan, leveren de nodes nog steeds een bijdrage aan het trainen van het model. Hierdoor wordt het probleem met vele ‘dode nodes’ verholpen, en werken deze nodes wel mee om zo hopelijk overfitting tegen te gaan. 

#### Specifieke probleem
Het huidige probleem is nog steeds het overfitten van het huidige model. De oplossing die in het huidige hoofdstuk wordt aangedragen zijn het wijzigen van de ReLu-activaties naar Leaky ReLu-activaties. De nodes die door ReLu-activaties dode nodes opleveren worden met Leaky ReLu-activaties wel actief, en helpen zo het model minder specifiek de trainingsdata fitten om zo minder te overfitten.

#### Veranderingen in het model
In elke van de vier Conv2D-lagen is de ReLu-activatie functie vervangen door een Leaky-ReLu functie. De alpha - de hoek van de Leaky-ReLu functie- is in verschillende versies gevarieerd om zo de optimale waarde te vinden.

![image](https://user-images.githubusercontent.com/68432564/150955536-92922c40-f1c4-40ee-8a21-e30cb425b1ee.png)

_Figuur 9:_ de ReLU en Leaky ReLU activaties 

De ReLU (links) neemt de input en veranderd elke negatieve input naar 0, maar houdt alle input groter dan 0. Uit het toepassen van ReLU komt echter het probleem dat de kans aanwezig is dat er dode nodes worden gecreëerd. Om dit te voorkomen wordt overgeschakeld naar Leaky ReLU (rechts). De Leaky ReLU functie staat in de afbeelding met een voorbeeld waarde van 0.01. Deze waarde geeft aan hoe steil de functie loopt wanneer x lager is dan 0. In dit model gaan we testen met Leaky ReLU waardes van , 0.1, 0.2 en 0.3. 

### Data Analyse en Voorverwerking
Bij deze verbeterstap is geen data analyse of voorverwerking van pas gekomen maar is de data gebruikt zoals in de vorige hoofdstukken.

### Model Pipeline en Training
In versie 1 van deze verbeterstap is de alpha van de Leaky-ReLu functies ingesteld op 0.1.

In versie 2 van deze verbeterstap is de alpha van de Leaky-ReLu functies verhoogd naar 0.2 om te kijken of dit een grotere validatie accuraatheid oplevert. 

In versie 3 van deze verbeterstap is de alpha van de Leaky-ReLu functies nogmaals verhoogd naar 0.3.

### Evaluatie en Conclusies

In figuur 10 zijn de resultaten te zien van de eerste versie met een alpha van 0.1. Opmerkelijk is dat de validatie accuraatheid een flinke sprong heeft gemaakt van 0.52 naar ongeveer 0.61. Wat dat betreft heeft deze nieuwe activatiefunctie een zeer positieve invloed gehad. Echter is de vorm van de grafieken niet zozeer aangepast en blijft de accuraatheid van de trainingsdata zowat gelijk, dus blijft het probleem van overfitting bestaan. Ook zijn de kosten voor validatie van het model gestegen.

![image](https://user-images.githubusercontent.com/68432564/150955651-8077f24b-d993-4a25-9b75-fed170c80f13.png)
![image](https://user-images.githubusercontent.com/68432564/150955689-a5e4f275-1fca-4495-a4cc-a4b4801cf78b.png)

_Figuur 10:_ Kosten en accuraatheid met een alpha van 0.1 voor de Leaky-ReLu activaties

In figuur 11 zijn de resultaten weergegeven wanneer de alpha is verhoogd naar 0.2. Hierbij kan niet echt iets worden gezegd over de toename of afname van de accuraatheid omdat die in eerste instantie afneemt naar 0.59 en in tweede instantie stijgt naar 0.62. Ook zijn de validatie kosten nog verder toegenomen.

![image](https://user-images.githubusercontent.com/68432564/150955866-dd1d3772-7816-4e10-b4eb-bbef34ab85fa.png)
![image](https://user-images.githubusercontent.com/68432564/150955923-d09a8283-4093-436d-b2d2-bcefb44c9c95.png)

_Figuur 11:_ Kosten en accuraatheid met een alpha van 0.2 voor de Leaky-ReLu activaties

In figuur 12 valt te zien dat de hogere alpha van 0.3 geen grotere validatie accuraatheid oplevert ten opzichte van een alpha van 0.1. De accuraatheid zit op 0.61 en 0.58 wat ongeveer hetzelfde resultaat is als de alpha van 0.2. Daarentegen zijn de validatie kosten wel weer verder toegenomen vergeleken bij een Leaky-Relu met een alpha van 0.1 en 0.2. Blijkbaar is de Leaky-ReLu functie hier dus te steil aflopend.

![image](https://user-images.githubusercontent.com/68432564/150956045-db8cf545-96fa-4ed7-b4d6-c25ea220f15d.png)
![image](https://user-images.githubusercontent.com/68432564/150956073-48b61a41-73ee-4619-95dc-655e45ff3587.png)

_Figuur 12:_ Kosten en accuraatheid met een alpha van 0.3 voor de Leaky-ReLu activaties

Na het testen van deze verschillende versies kan geconcludeerd worden dat een Leaky-ReLu activatie met een alpha van 0.1 de beste aanvulling vormt op ons basismodel. De validatie accuraatheid is daarmee iets hoger en dat is waar naar gestreefd dient te worden. Bovendien is dit verreweg het voordeligst kijkend naar de kosten. De kosten van een alpha van 0.3 zitten net boven de 100 terwijl de kosten van een alpha van 0.1 rond de 20 liggen. 

Wanneer gebruik wordt gemaakt van (Leaky) ReLu activatie functies is het van belang dat deze non-lineair zijn om het netwerk diep te kunnen laten leren. Als de functies lineair zijn is de capaciteit van het netwerk om te leren niet optimaal. Een methode om er zeker van te zijn dat de Leaky ReLu activatie functies non-lineair zijn is het normaliseren van de trainingsdata voordat deze aan het netwerk wordt meegegeven.


## 6. Normaliseren van de afbeeldingen
### Introductie
Door de input data te normaliseren krijgt de input data een gemiddelde van 0. Leaky ReLu activatie functies zijn niet-lineair als de input ervan 0 is, en bij een neuraal netwerk dat gebruik maakt van (Leaky) ReLu activatie functies zoals de huidige is het van belang dat de lagen niet-lineair zijn om het netwerk diep te laten leren. Een model kan een voorspelling maken op basis van de activatie functies. Wanneer een lineaire activatie functie wordt meegegeven zal het model dus alleen een lineaire functie kunnen gebruiken om tot een voorspelling te komen. Deze voorspellingen zullen een stuk minder accuraat zijn dan wanneer een non-lineaire functie wordt gebruikt omdat deze functie een stuk complexere vormen aan kan nemen. 

Het normaliseren van de input data heeft daarom als effect dat de Leaky ReLu activatie functies beter kunnen functioneren dan wanneer de input data niet genormaliseerd wordt. Een methode om de input data (training afbeeldingen) te normaliseren is door gebruik te maken van de ImageDataGenerator preprocesser tool (2). Deze methode functioneert door over de gehele dataset de afbeeldingen te normaliseren naar een gemiddelde van 0 en een standaarddeviatie van 1. Dit houdt in dat bijvoorbeeld een vaag gekleurde afbeelding met lage pixelwaardes evenveel meegerekend wordt als een afbeelding met hele felle kleuren. Verder helpt data preprocessen het model sneller te maken.

#### Specifieke probleem
Om de Leaky ReLu activatie functies zo goed mogelijk te laten functioneren is het van belang dat de input data genormaliseerd wordt. Wanneer dit niet wordt gedaan leert het netwerk minder diep vanwege dode nodes. 

#### Veranderingen in het model
In de functie die wordt het gebruikt om het model te trainen en evalueren (train_en_evalueer) is het argument ‘preprocess’ toegevoegd met parameters ‘featurewise center’ en ‘featurewise standard deviation’ uit de keras bibliotheek.

### Data Analyse en Voorverwerking
Alle data is dezelfde data die gebruikt werden in in de voorgaande netwerkversies. Er zijn dus geen aanpassingen aan de data gedaan. 

### Model Pipeline en Training
Het preprocess argument dat mee wordt genomen tijdens het trainen en evalueren van het netwerk transformeert de training afbeeldingen op de manieren die zijn meegegeven in de specifiek opgegeven parameters uit de ImageDataGenerator bibliotheek van keras. In deze versie zal gekeken worden naar het gebruik van ‘featurewise center’ en ‘featurewise standard deviation (hierna std) normalization’. ‘Featurewise’ center wijzigt de gemiddelde waarde van de inputdata naar 0. ‘Featurewise standard deviation normalization’ deelt elke inputwaarde door de bijbehorende standaarddeviatie waardoor de data een verdeling krijgt tussen de waardes -1 en 1. 

### Evaluatie en Conclusies
In figuur 13 is te zien dat, ten opzichte van figuur 10, 11 en 12, de accuraatheid van de validatie data met ongeveer 10 procent is gestegen. Hierdoor kan worden gesteld dat het toevoegen van de preprocess technieken een positief resultaat heeft op de accuraatheid van het netwerk.

![image](https://user-images.githubusercontent.com/68432564/150956269-ee1d59ce-0ce9-433d-8783-221e89a7452e.png)

_Figuur 13:_ Kosten en accuraatheid bij toepassen van normalisatie.

Na gezien te hebben dat de Leaky ReLU goed is geïmplementeerd, kan weer worden gekeken naar het overfitten. Er is wellicht niet genoeg (verschillende) data in sommige klassen waardoor die klassen heel lastig zijn om goed te voorspellen. Om dit op te lossen kan worden gekeken naar de klassen die het slechtst worden voorspeld. Het doel is om te achterhalen waarom deze klassen zo slecht worden voorspeld. Wellicht bestaan deze klassen uit te weinig data om het netwerk goed te trainen en kan het netwerk hierdoor bepaalde features van de vogelsoorten niet goed leren.  

## 7. Data analyse 
### Introductie
Het verkeerd kwalificeren van soorten kan liggen aan de trainingsdata die aan het model gevoed wordt. Een manier om zowel het overfitten tegen te gaan, als de validatie accuraatheid te verhogen, is te kijken naar de verdeling van de data. Door middel van barplots is hier inzicht in te verkrijgen. In een barplot valt namelijk te zien hoe de data verdeeld is. Wanneer de data ongebalanceerd verdeeld is, kan het netwerk bepaalde vogelsoorten beter leren herkennen en daardoor beter classificeren dan andere vogelsoorten. 

#### Specifieke probleem
Het netwerk overfit nog steeds. Het is mogelijk dat het netwerk beter wordt getraind op sommige soorten dan anderen omdat er voor deze soorten meer trainingsdata beschikbaar is.

#### Veranderingen in het model
Om dit probleem tegen te gaan wordt een data analyse uitgevoerd. Het model wordt hierbij niet aangepast, maar er wordt een analyse op de trainingsdata uitgevoerd waarbij wordt gekeken uit hoeveel afbeeldingen de soorten in de trainingsdata bestaan. 

### Data Analyse en Voorverwerking
Voor de data analyse is een functie geschreven genaamd data_grafiek. Deze functie neemt als input de training labels en geeft als uitput de 5 meest voorkomende en 5 minst voorkomende vogelsoorten weer.

### Model Pipeline en Training
Het netwerk is gelijk gebleven aan het netwerk in het voorgaande hoofdstuk.

### Evaluatie en Conclusies
Uit de barplots blijkt dat de trainingsdata ongebalanceerd verdeeld is; 
De vogelsoorten ‘SPOTTED CATBIRD’ (116), ‘CASSOWARY’ (119), ‘BLACK SWAN’ (119), ‘INCA TERN’ (119) en ‘BARN OWL’ (120) komen bij deze dataset het minst voor, waarbij de ‘SPOTTER CATBIRD’ het minst vaak voorkomt (116 keer). 
De vogelsoorten ‘HOUSE FINCH’ (249), 'OVENBIRD' (233), 'D-ARNAUDS BARBET' (233), 'SWINHOES PHEASANT' (217), 'WOOD DUCK' (214) komen het meest voor, waarbij de ‘HOUSE FINCHhet meest voorkomt (249 keer). De meest voorkomende vogelsoort komt dus meer dan twee keer zo veel voor in de dataset dan de minst voorkomende vogelsoort.

![image](https://user-images.githubusercontent.com/68432564/150956410-87485061-b37d-4ece-8cd1-77f8d6b9da0e.png)

_Figuur 14:_ Overzicht van de hoeveelheid trainingsafbeeldingen per vogelsoort.

De trainingsdata bevat van sommige vogelsoorten een stuk meer afbeeldingen dan van andere vogelsoorten. Dit zou er mogelijk toe kunnen leiden dat vogelsoorten met weinig afbeeldingen slechter voorspeld worden door het netwerk omdat hier weinig data voorhanden is, en daarmee bijdragen aan een lage accuraatheid. 

Een volgende stap is het toepassen van een methode die ervoor zorgt dat de vogelsoorten met minder trainingsafbeeldingen net zo veel mee worden genomen in het trainen van het netwerk als de vogelsoorten die uit meerdere trainingsafbeeldingen bestaan. Hier zijn meerdere opties voor. 
Een optie is de verdeling van de trainingsafbeeldingen per vogelsoort gelijk te trekken door van elke vogelsoort 116 trainingsafbeeldingen aan te houden en de extra trainingsafbeeldingen die beschikbaar zijn per vogelsoort te verwijderen uit de dataset. Een groot nadeel hiervan is dat veel goed te gebruiken data om het netwerk te trainen verloren gaat.
Een tweede optie is het toevoegen van extra inputdata alleen op de vogelsoorten waar weinnig afbeeldingen van zijn om op deze manier de hoeveelheid trainingsafbeeldingen per vogelsoort gelijk te trekken. Het is lastig om nieuwe afbeeldingen van deze vogelsoorten te vinden en deze nog extra in te laden. Er zou voor kunnen worden gekozen om door middel van data augmentatie extra afbeeldingen van deze vogelsoorten in te voegen. Er is echter een optie die nog makkelijker en even doeltreffend zou moeten zijn.
Deze laatste optie is om bij het trainen van het netwerk mee te geven aan het netwerk dat de vogelsoorten die uit minder trainingsafbeeldingen bestaan zwaarder mee worden genomen in het trainen van het netwerk dan vogelsoorten met meer trainingsafbeeldingen. Deze methode wordt in het hiernavolgende hoofdstuk toegepast op het netwerk. 


## 8. Data augmentatie (weighted)
### Introductie
De trainingsdata is ongebalanceerd verdeeld, van sommige vogelsoorten bestaan namelijk bijna twee keer zoveel trainings afbeeldingen dan bij andere vogelsoorten. Een manier om de disbalans in de trainingsdata op te lossen is om de soorten die slecht vertegenwoordigd zijn in de trainingsdata zwaarder mee te laten wegen bij het trainen van het netwerk en oververtegenwoordigde soorten lichter mee te laten wegen. Hierdoor wordt het model diverser getraind en heeft het netwerk zo minder de neiging tot overfitten en is het netwerk meer generaliseerbaar naar de werkelijkheid. De testdata en de validatie data zijn vergeleken met de training data wel gebalanceerd verdeeld. Van elke vogelsoort zijn er bij deze data namelijk 5 afbeeldingen aanwezig.

#### Specifieke probleem
De trainingsdata bevat van sommige vogelsoorten een stuk meer afbeeldingen dan van andere vogelsoorten. Een mogelijk gevolg is dat vogelsoorten met weinig afbeeldingen slechter voorspeld worden door het netwerk en daarmee bijdragen aan een lage accuraatheid. 

#### Veranderingen in het model
In het model zelf is niets aangepast. Het wegen van de data wordt tijdens het fitten van het model toegepast.

### Data Analyse en Voorverwerking
De afbeeldingen van soorten moeten worden gewogen op basis van frequentie. Dit is geïmplementeerd door de gewichten van de soorten op basis van frequentie ten opzichte van alle andere soorten op te slaan als values per soort in een dictionary. 

### Model Pipeline en Training
Op het moment dat het model de verwachte waarden fit, wordt de dictionary ingevoerd bij het argument class_weights. Hierdoor krijgt het netwerk tijdens het fitten de boodschap mee dat het bij het berekenen van de kostenfunctie meer rekening moet houden met weinig voorkomende vogelsoorten, en minder met veel voorkomende vogelsoorten. 


### Evaluatie en Conclusies
Op basis van de kosten en accuraatheid weergeven in figuur 14 valt te concluderen dat het zwaarder mee laten wegen van ondervertegenwoordigde trainingsdata en het lichter mee laten wegen van oververtegenwoordigde trainingsdata in de kostenfunctie helaas niet heeft geleid tot een beter voorspellend model; sterker nog, het model voorspelt nu slechter dan voor deze aanpassing. Weliswaar komt de kostenfunctie nog wel overeen met de eerdere situatie, maar is de validatie accuraatheid teruggelopen van rond de 0.7 naar 0.65. Hoewel de trainingsdata nu verschillend meewegen in de kostenfunctie aan de hand van hoe de trainingsdata verdeeld zijn, voorspelt het model nu slechter door het zwaarder wegen van de soorten die weinig voorkwamen. Een conclusie die hieraan te verbinden valt is dat de specifieke vogelsoorten die weinig voorkomen in de trainingsdata lastiger te voorspellen zijn dan vogelsoorten die veel voorkomen in de trainingsdata. Door deze data dan zwaarder mee te laten wegen neemt de algehele validatie accuraatheid af.

![image](https://user-images.githubusercontent.com/68432564/150956494-753b1ecc-ecb2-45fb-a53c-90df5a6d2852.png)

_Figuur 15:_ Kosten en accuraatheid van het model waarbij ondervertegenwoordigde trainingsdata zwaarder meewegen in de kostenfunctie, en oververtegenwoordigde trainingsdata minder zwaar meewegen.

Omdat het verschillend mee laten wegen van de trainingsdata in de kostenfunctie geen effect oplevert, lijkt het ons verstandig om in de volgende vervolgstap te onderzoeken bij welke data het netwerk het meest de fout in gaat. De voorspelling die volgt uit het huidige hoofdstuk is dan dat de vogelsoorten die uit minder trainingsafbeeldingen bestaan het lastigst te classificeren zijn. Om deze hypothese te testen is een goede vervolgstap om te analyseren bij welke vogelsoorten de classificatie niet correct verloopt. De bevindingen die hieruit worden opgedaan kunnen verder worden verwerkt en geprobeerd tegen te gaan door passende data augmentatie toe te passen. 

MILESTONE 4 deel 1
## 9. Netwerk analyse
### Introductie
Om het overfitten van het netwerk verder tegen te gaan en de validatie accuratie van het model te verhogen, kan analyseren van hoe het netwerk tot nu toe presteert per vogelsoort relevante informatie opleveren. Door te kijken naar de validatie accuracy per vogelsoort, kan in kaart worden gebracht welke vogelsoorten het netwerk voornamelijk correct weet te voorspellen en welke niet. Aan de hand van deze resultaten kunnen wellicht overeenkomsten gevonden worden tussen vogelsoorten die wel goed voorspeld worden door het netwerk en vogelsoorten die niet goed worden voorspeld door het netwerk. Door te weten welke aspecten ervoor zorgen dat het netwerk een vogelsoort wel of juist niet correct voorspelt kan er worden geprobeerd dit aspect meer naar voren te laten komen of om hier juist voor te corrigeren. 

Het analyseren van de accuraatheid per vogelsoort kan worden gedaan aan de hand van een confusion matrix. In een confusion matrix staan de accuraties per vogelsoort namelijk op de diagonaal.  Aan de hand van de accuraatheden op de diagonaal kunnen de vogelsoorten geselecteerd worden met de laagste en hoogste accuraatheid. Aan de hand van de informatie verkregen uit de accuraatheden, kan bepaald worden op welke manieren verdere augmentatie van de data van pas zou komen. 

#### Specifieke probleem
Het huidige netwerk overfit nog steeds en de validatie accuratie is nog niet optimaal. 

#### Veranderingen in het model
Er zijn geen veranderingen aangebracht aan het netwerk. 

### Data Analyse en Voorverwerking
Een confusion matrix geeft een duidelijk overzicht van hoe goed en hoe slecht de data voorspeld wordt met een classificatie model. Op zowel de x-as als de y-as staan de vogelsoorten, waarbij voor een specifieke vogelsoort de index van de rij hetzelfde is als de index van de column (dus de albatross staat bijvoorbeeld op zowel rij 1 als column 1). De x-as verwijst vervolgens naar de ware vogelsoort, en de y-as naar de voorspelde vogelsoort. Vanwege deze indeling, staan op de diagonaal de accuraatheden van elke vogelsoort. Hierbij houdt de accuraatheid in hoe groot de ratio is van vogels die tot een bepaalde soort behoren en door het netwerk ook tot deze soort zijn geclassificeerd. Aan de hand van de accuraatheden op de diagonaal kunnen de vogelsoorten geselecteerd worden met de laagste en hoogste accuraatheden.

### Model Pipeline en Training
Het netwerk hoefde niet gerund te worden voor dit hoofdstuk. 

### Evaluatie en Conclusies
De vogelsoorten met de 5 laagste en 5 hoogste accuraatheden zijn als volgt.
De 5 vogelsoorten die het slechtst herkend worden zijn: 'PHILIPPINE EAGLE', 'ELLIOTS  PHEASANT', 'GRAY PARTRIDGE', 'DARK EYED JUNCO' en de 'STRAWBERRY FINCH'. 
Hieronder volgen deze vogelsoorten met 3 bijbehorende afbeeldingen.

![image](https://user-images.githubusercontent.com/68432564/150956669-5cbc20a7-5219-4dca-aaa8-b71e44c7b206.png)

_Afbeelding 2:_ Drie afbeeldingen van de Philippine Eagle.

![image](https://user-images.githubusercontent.com/68432564/150956781-6d8970ed-0cb6-484a-9517-ac2b195385f0.png)

_Afbeelding 3:_ Drie afbeeldingen van de Elliots Pheasant.

![image](https://user-images.githubusercontent.com/68432564/150956831-98e86d44-8185-41cf-adec-bbb6fb48a61b.png)

_Afbeelding 4:_ Drie afbeeldingen van de Gray Partridge.

![image](https://user-images.githubusercontent.com/68432564/150956905-b7de6b23-e28c-455f-8a65-c1b3035a73da.png)

_Afbeelding 5:_ Drie afbeeldingen van de Dark Eyed Junco.

![image](https://user-images.githubusercontent.com/68432564/150956961-600dd67a-60b1-4eab-bc25-fb0cfb0d31e4.png)

_Afbeelding 6:_ Drie afbeeldingen van de Strawberry Finch.

De 5 vogelsoorten die het best herkend worden zijn: 'STRIPPED MANAKIN', 'AFRICAN FIREFINCH', 'SNOWY EGRET', 'GOLD WING WARBLER’ en de 'RED NAPED TROGON’. Hieronder volgen deze vogelsoorten met 3 bijbehorende afbeeldingen.

![image](https://user-images.githubusercontent.com/68432564/150957035-93d8ccc8-5c46-4a97-a4bc-e91a446caeea.png)

_Afbeelding 7:_ Drie afbeeldingen van de Stripped Manakin.

![image](https://user-images.githubusercontent.com/68432564/150957110-fa6bcb30-4281-4a3e-b66d-76288fd3cc18.png)

_Afbeelding 8:_ Drie afbeeldingen van de African Firefinch.

![image](https://user-images.githubusercontent.com/68432564/150957171-22f09707-9e68-4bbb-9dfa-c1a0721ae11c.png)

_Afbeelding 9:_ Drie afbeeldingen van de Snowy Egret.

![image](https://user-images.githubusercontent.com/68432564/150957226-dbdf148c-4a8f-4bc5-81ed-1942e6ac9238.png)

_Afbeelding 10:_ Drie afbeeldingen van de Gold Wing Warbler.

![image](https://user-images.githubusercontent.com/68432564/150957274-d9dd6594-cdb7-4d45-8ca4-432716704769.png)

_Afbeelding 11:_ Drie afbeeldingen van de Red Naped Trogon.

Op te maken uit de afbeeldingen van de slechtst herkenbare en best herkenbare vogelsoorten is voornamelijk het verschil in kleur. De vogels die het vaakst correct worden geclassificeerd zijn vogels met opvallende kleuren. De vogels die het minst vaak correct worden geclassificeerd daarentegen zijn vogels met onopvallende kleuren. Daarnaast lijken deze onopvallende kleuren meer op de achtergronden waarop de vogels afgebeeld zijn. De vogels die het slechts worden geclassificeerd lijken hierdoor meer weg te vallen. Om het netwerk beter te trainen zijn twee opties mogelijk; ofwel het vermeerderen van de trainingsdata ofwel de beschikbare trainingsdata zo bewerken dat het netwerk meer van de data leert. Het vermeerderen van de data door middel van data augmentatie en het verwerken van de bevindingen van dit hoofdstuk door de bestaande trainingsafbeeldingen te bewerken zal in de komende twee hoofdstukken één voor één behandeld worden.

MILESTONE 4 deel 2
## 10. Data augmentatie (horizontale flip)

### Introductie
Uit de netwerk analyse bleek dat sommige vogelsoorten nog niet erg goed gekwalificeerd kunnen worden door het model op basis van de huidige trainingsdata. Dit probleem valt deels te verhelpen door een data augmentatie op de trainingsdata. Door bij het preprocessen van de afbeeldingen ook de foto’s te spiegelen en deze gespiegelde foto’s extra toe te voegen aan de trainingsdata, is er meer data en ook meer diverse data voorhanden voor het model om van te leren. Het model zou dus met hogere accuraatheid moeten kunnen voorspellen na deze alteratie. Daarnaast valt zo ruis weg van de trainingsdata omdat hele specifieke aspecten van de afbeeldingen ook gespiegeld worden en zo verzwakt worden, dus de overfitting wordt wederom aangepakt. 
Bij het kiezen van de as van spiegelen ligt horizontaal meer voor de hand. Na bestudering van enkele foto’s uit de dataset werd ons duidelijk dat de vogels zeer vaak in gewone rechtopstaande positie afgebeeld zijn, en niet bijvoorbeeld op de kop hangend. Door deze rechte afbeeldingen horizontaal te spiegelen ontstaan nieuwe afbeeldingen van ook rechtopstaande vogels, maar dan de andere kant op gedraaid. Zo zouden de vogels ook in nieuwe sets van data zoals de validatie data kunnen voorkomen. Een verticale spiegeling ligt niet voor de hand, aangezien er weinig vogels in onze trainingsdata op de kop hangend zijn afgebeeld, en dit verticaal spiegelen ook niet een herkenbaar beeld zou opleveren. 

#### Specifieke probleem
In het huidige model worden sommige vogelsoorten nog niet accuraat genoeg door het model voorspeld. Door de afbeeldingen te spiegelen en ook op te nemen in de trainingsdata wordt de trainingsdata meer divers en omvangrijk, en kan overfitting tegengegaan worden.

#### Veranderingen in het model 
Het model berekent nog op eenzelfde manier het definitieve model, er zijn slechts aanpassingen gedaan aan de trainingsdata.

### Data Analyse en Voorverwerking
Het spiegelen van afbeeldingen is geïmplementeerd binnen de preprocess stap. Hierin is horizontal-flip op True gezet. Dit houdt in dat random wordt bepaald voor een afbeelding in de trainingsdata of deze horizontaal gespiegeld aan de trainingsdata wordt toegevoegd. 

### Model Pipeline en Training
De aanpassing van dit hoofdstuk heeft geen invloed op de Model Pipeline

### Evaluatie en Conclusies
Bestudering van de resultaten van dit hoofdstuk in figuur 16 leidt tot de conclusie dat er mooie progressie is geboekt met toevoeging van de horizontale spiegelingen in de trainingsdata. Gelijk valt op dat de validatie accuraatheid is gestegen van 0.7 naar 0.75. Ook zijn de validatie kosten gedaald en zijn de lijnen van training- en validatie accuraatheid dichterbij elkaar komen te liggen. Hieruit valt op te maken dat overfitten weer deels is verholpen. Dit zijn allemaal gewenste resultaten. De keerzijde is wel dat het model langzamer is geworden, vanwege de toevoeging van de gespiegelde afbeeldingen aan de trainingsdata. Het kost nu meer tijd om deze grotere trainingsset te verwerken. Aangezien deze augmentatie zich slechts richtte op het verkrijgen van meer trainingsdata - een vrij algemene stap naar verbetering bij het bouwen van een model - kan op basis van de diepgaande data-analyse ook nog gekeken worden naar augmentaties die specifiek gericht zijn op vogelsoorten die slecht voorspeld zijn.

![image](https://user-images.githubusercontent.com/68432564/150957358-65119ad2-55aa-4dff-bded-857c88b0b0b5.png)

_Figuur 16:_ Kosten en accuraatheid bij horizontaal gespiegelde afbeeldingen in trainingsdata.

MILESTONE 4 deel 3
## 11. Data augmentatie (croppen)
### Introductie
Uit de netwerk analyse is gebleken dat voor een aantal klassen de vogelsoorten niet goed herkend kunnen worden. Dit kan komen doordat bv. de kleuren van de vogels te veel overeenkomen met de kleuren in de achtergrond. Voor slechts een paar afbeeldingen hoeft dit geen probleem te zijn, maar als er bijvoorbeeld in de dataset veel afbeeldingen voorkomen van bruine vogels in hun bruine natuurlijke leefomgeving kan het model hier snel moeite mee krijgen. Om dit te voorkomen kan worden gekeken of het inzoomen op de foto’s tot een verbetering leidt. Zo wordt de focus voor het model meer gelegd op de essentiële onderscheidende informatie van de vogel die zich meer in het centrum van de afbeelding bevindt.

#### Specifieke probleem
Het model kan nu nog erg slecht bepaalde vogelsoorten herkennen. Deze vogelsoorten hebben voornamelijk ruis van de achtergrond omdat de vogels en de achtergrond vaak (bijna) dezelfde kleur hebben. 

#### Veranderingen in het model
Na het opbouwen van het model kan in het preprocess argument een nieuwe techniek worden meegegeven. Om te kunnen croppen wordt de techniek ‘zoom range’ meegegeven. Deze techniek zoomt in op het midden van de foto met een bepaalde factor. Zo zoomt een factor 0.5 2 keer in op het midden van de foto. 

### Data Analyse en Voorverwerking
Aan deze verbeterstap is geen data analyse of voorverwerking van pas gekomen maar is de data gebruikt zoals in de vorige hoofdstukken.

### Model Pipeline en Training
Om het model te trainen met de ingezoomde data wordt het preprocess argument in de train en evalueer functie aangepast. Hier wordt de ‘zoom range’ techniek toegevoegd. Deze techniek zoomt in op het midden van de foto met een bepaalde factor of bereik. Als hier een factor wordt ingevoerd wordt er zowel in als uitgezoomd met deze factor. Bij een zoomfactor van 0.25 wordt 25% ingezoomd zodat er wordt gekeken naar 75% van de foto en wordt 25% uitgezoomd naar 125%. Wij willen echter niet naar het uitgezoomde deel van de foto kijken, aangezien is vastgesteld dat de essentiële informatie zich meer in het centrum van de afbeelding bevindt. Daarom gebruiken wij een bereik. Met het bereik kan een lijst met 2 waardes worden meegegeven. De foto wordt ingezoomd op basis van de opgegeven waardes. Een lijst met 0.75, 1.1 zal ingezoomd worden van 75% tot uitzoomen van 10%. In dit model wordt gekeken naar de bereiken van [0.75, 1] en [0.5, 0.75]. De eerste zoom zal een kleine zoom doorvoeren en zo een beter beeld van de vogel geven. De tweede zoom probeert meer focus te leggen op de vogel en probeert echt de ruis in de achtergrond weg te snijden. 

### Evaluatie en Conclusies
De zoom bereiken waar naar wordt gekeken zijn [0.75, 1] en [0.5, 0.75]. Het resultaat van de eerste zoom range staat afgebeeld in figuur 17. Te zien is dat de accuraatheid van het model omhoog is gegaan naar 77%. Dit is een verbetering van 3% t.o.v. het invoegen van de horizontal flip. 

![image](https://user-images.githubusercontent.com/68432564/150957528-20f48b34-7c49-4e69-b56c-9e30d39cac0a.png)

_Figuur 17:_ Kosten en accuraatheid bij een zoom range van [0.75, 1]

Naast het kijken naar een zoom range van [0.75, 1] willen we ook zien of er meer op de vogels gefocust kan worden. Deze zoom gaf echter minder succes dan de eerdere zoom. De zoom van [0.5, 0.75] staat afgebeeld in figuur 18. Te zien is dat er een grote hap is genomen uit de accuraatheid. De accuraatheid ligt op 68% terwijl deze hiervoor naar 77% was gebracht.

![image](https://user-images.githubusercontent.com/68432564/150957637-613e6fa7-81b7-4ddb-ae77-5c76b8f5db14.png)

_Figuur 18:_ Kosten en accuraatheid bij een zoom range van [0.5, 0.75]

De verbetering van dit hoofdstuk vormt het slotstuk van de data augmentatie; wij hebben nu enkele methoden uitgeprobeerd en zijn tevreden met de geboekte resultaten. Echter is er nog voldoende progressie te boeken, en vindt er ook zeker nog overfitting plaats. Daarom gaan wij in het volgende hoofdstuk weer verder met nieuwe methodes om overfitting tegen te gaan.

MILESTONE 4 deel 4

## 12. Drop-out methode op verdiept netwerk

### Introductie
Een welbekende methode tegen overfitting die ook al eerder in dit verslag toegepast is, is Dropout. In deze eerdere poging bleek dat het model nog niet complex genoeg was om Dropout toe te passen; in dat geval zorgde de Dropout voor het verlies van te veel essentiële informatie, waardoor de validatie accuraatheid afnam. Aangezien het huidige model wel meerdere lagen bevat, lijkt het dat Dropout nu wel tot een vermindering van overfitting kan leiden. 

#### Specifieke probleem
Het probleem dat getackeld dient te worden met deze ingreep is het overfitten van de trainingsdata door het model. Door op basis van kans delen van de input in bepaalde epochs niet mee te nemen wordt de invloed van input die veel ruis veroorzaakt verkleind.

#### Veranderingen in het model
Wederom zijn er drie versies van Dropout toegepast, om zo de optimale toepassing te vinden. 

In de eerste versie zijn de meeste Dropout-lagen toegevoegd, drie om precies te zijn. 

In versie twee is de laatste Dropout-laag verwijderd, om te kijken of dit tot een beter resultaat leidt. Er is gekozen om de laatste van de drie lagen te verwijderen, omdat op dit moment het model al vrij vergaand getraind is, en Dropout op dit moment eerder zou leiden tot verlies van informatie. 

In de laatste versie, versie drie, is de tweede Dropout-laag ook verwijderd en blijft slechts de eerste laag over. 

### Data Analyse en Voorverwerking
Bij deze verbeterstap is geen data analyse of voorverwerking van pas gekomen maar is de data gebruikt zoals in de vorige hoofdstukken.

### Model Pipeline en Training
Het voorgaande model is aangevuld met drie Dropout-lagen, die telkens na de MaxPooling-lagen geplaatst zijn. De waarschijnlijkheid in deze lagen is gesteld op 0.2, een redelijk lage waarde die niet tot al te extreme Dropout zal leiden. In figuur 19 is te zien hoe dit voor versie 1 eruit komt te zien. In de latere versies zijn slechts Dropout-lagen hieruit verwijderd, dus dit geeft een goede impressie.

![image](https://user-images.githubusercontent.com/68432564/151187282-1caff67c-7959-4f51-940a-46d16f775647.png)

_Figuur 19:_ Model-summary bij toevoeging van de drie Dropout-lagen (versie 1)

### Evaluatie en Conclusies

In figuur 20 zijn de bevindingen van versie 1 weergegeven. Te zien valt dat de overfitting is tegengegaan door toevoeging van de drie Dropout-lagen. De grafieken van training en validatie liggen voor zowel de kosten als accuraatheid dichterbij elkaar; de accuraatheid van de validatie data is niet gestegen maar nog exact hetzelfde, maar de accuraatheid van de trainingsdata is wel gedaald. Ook verloopt de asymptoot van de accuraatheid van de trainingsdata minder steil en haalt deze nu niet meer de 0.9, wat inhoudt dat het model minder snel gewend raakt aan de specifieke ruis van de trainingsdata. Daarnaast valt op dat de kosten voor de validatie data een minder sterk stijgende lijn vertonen, wat betekent dat deze minder oploopt naarmate het model meer gaat overfitten op de trainingsdata.

![image](https://user-images.githubusercontent.com/68432564/151187536-7e3b3f8f-2ff7-4f08-bf2d-f2ffeedd0e16.png)

_Figuur 20:_ Kosten en accuraatheid bij drie Dropout-lagen van 0.2.

In figuur 21 zijn de resultaten van versie twee weergegeven. Hieruit blijkt dat het weglaten van de laatste Dropout-laag vrij weinig invloed heeft op de vorm van de grafieken. Wel valt op dat de accuraatheid van de validatie data is gedaald met 0.03 tot 0.74. 

![image](https://user-images.githubusercontent.com/68432564/151187628-246b630d-c0c9-4a53-a8fe-af5fd0784f21.png)

_Figuur 21:_ Kosten en accuraatheid bij twee Dropout-lagen van 0.2.

In figuur 21 zijn de resultaten van versie twee weergegeven. Dit is duidelijk de meest milde variant van Dropout die wij hebben toegepast. Hoewel overfitting van het basismodel is tegengegaan, zijn nog duidelijk wel kenmerken van de beginsituatie te herkennen; zo is de lijn van de validatie kosten nog steeds stijgend, en bereikt de accuraatheid van de training weer de 0.9. 

![image](https://user-images.githubusercontent.com/68432564/151187705-a47485d9-1e65-4550-8523-b652f372bdb8.png)

_Figuur 22:_ Kosten en accuraatheid bij één Dropout-laag van 0.2.

Concluderend lijkt versie 1 die de meest ingrijpende toepassing van Dropout bevat de beste aanvulling op ons basismodel. Bij deze versie wordt overfitting tegengegaan, terwijl de validatie accuraatheid precies gelijk blijft. Ten opzichte van het basismodel waar versie 2 en 3 een uitbreiding op vormen, zouden dit wel verbeteringen zijn omdat de overfitting wordt tegengegaan terwijl er maar een marginaal verlies op de validatie accuraatheid wordt geleden. Echter biedt versie 1 het beste van beide werelden: overfitting wordt tegengegaan terwijl dit niet ten koste gaat van de validatie accuraatheid. 

MILESTONE 4 deel 5

## 13. Batchnormalisatie

### Introductie
Om het overfitten van het model nog verder tegen te gaan bestaat er nog een methode die hiervoor gebruikt kan worden. Deze methode werd in hoofdstuk 2 bij het basismodel benoemd en uitgelegd, namelijk batchnormalisatie. 
Nogmaals in het kort; Bij batchnormalisatie wordt de input van de kanalen genormaliseerd door deze te centreren en te schalen aan de hand van aanleerbare parameters (gamma en beta). Batchnormalisatie kan helpen bij overfitting door het verschil in distributie van de input tussen verschillende datasets te reduceren, covariate shift genoemd. Om dezelfde distributie van data te behouden, worden de waardes van de input steeds genormaliseerd en neemt de accuratie van het model toe. Ook leert elke laag hierdoor de weights zelfstandiger aan. Naast dat batchnormalisatie helpt bij covariate shift, regulariseert het ook lichtelijk. Dit omdat door batchnormalisatie elke laag een zekere mate van ruis krijgt. Door ruis toe te voegen wordt het netwerk robuuster, minder ingesteld op de trainingsdata en simpeler, waardoor het model wellicht minder snel overfit. Het effect van overfitten tegengaan is niet een gegeven bij batchnormalisatie dus het hangt af van het huidige netwerk of het gewenste effect van minder overfitten door middel van batchnormalisatie zal worden bereikt.

#### Specifieke probleem
Het netwerk overfit nog steeds. De methode batchnormalisatie wordt toegepast om dit tegen te gaan.

#### Veranderingen in het model
In het netwerk wordt bij op kanaal batchnormalisatie toegepast. 

### Data Analyse en Voorverwerking
Er zijn geen wijzigingen aangebracht in de data die voor als eerste input gebruikt wordt. Het aanpassen van de data gebeurt tijdens het trainen van het model.

### Model Pipeline en Training
Op elk kanaal wordt de batchnormalisatie toegepast. De input van alle kanalen wordt genormaliseerd door deze te centreren en te schalen aan de hand van de aanleerbare parameters gamma en beta. Deze parameters worden geüpdate door middel van gradient descent. 

### Evaluatie en Conclusies
Een aantal verschillende hoeveelheden van batchnormalisatie zijn toegepast op het netwerk. Batchnormalisatie is toegepast op ofwel alle vier de kanalen, ofwel drie ofwel twee. Uit de resultaten is op te maken dat batchnormalisatie de accuratie validatie laat toenemen van rond de 77% tot rond de 80% en dat deze versies niet veel van elkaar verschillen. Wat duidelijk merkzaam is is dat de lijn die de validatie accuratie weergeeft nog verder lijkt te stijgen na het runnen van 20 epochs. 


![image](https://user-images.githubusercontent.com/68432564/151187947-6b4c24d8-9bc5-4f41-a991-2619d789aad2.png)

_Figuur 23:_ Netwerk met vier batchnormalisatie lagen.

![image](https://user-images.githubusercontent.com/68432564/151188088-5c33cb41-2c68-4731-a92a-f3c07964ddf2.png)

_Figuur 24:_ Netwerk met drie batchnormalisatie lagen.

![image](https://user-images.githubusercontent.com/68432564/151188167-72ebf271-5072-4f09-8931-0acdc33ebe5e.png)

_Figuur 25:_ Netwerk met twee batchnormalisatie lagen.

Het is dus mogelijk dat de validatie accuratie verder toeneemt wanneer er meer epochs worden gerund waardoor het netwerk langer kan worden getraind. Het uiteindelijke model dat wordt gekozen om te finaliseren is het netwerk waarbij op drie kanalen batchnormalisatie is toegepast. Hiervoor is gekozen omdat bij dit netwerk het meest duidelijk te zien is dat de validatie accuratie nog verder zou kunnen stijgen. Dit netwerk zal in het volgende hoofdstuk voor 200 epochs worden gerund om te kijken of de validatie accuratie nog verder toeneemt.

Bronnen:

1). https://towardsdatascience.com/analyzing-different-types-of-activation-functions-in-neural-networks-which-one-to-prefer-e11649256209

2). https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

3). https://www.kaggle.com/gpiosenka/100-bird-species
