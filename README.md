# Kreditkarten-Betrugserkennung

##Inhaltsverzeichnis
  - [Projektübersicht](#Projektübersicht)
  - [Ziele](#Ziele)
  - [Technologien_und_Werkzeuge](#Technologien_und_Werkzeuge)
  - [Daten](#Daten)
  - [Modellierung](#Modellierung)
  - [Ablauf](#Ablauf)
  - [Ergbnis](#Ergbnis)

## Projektübersicht
Dieses Projekt zielt darauf ab, betrügerische Transaktionen in Kreditkartendaten unter Verwendung fortschrittlicher Machine Learning-Techniken zu identifizieren. Der verwendete Datensatz besteht aus Transaktionen von europäischen Karteninhabern im September 2013, die mithilfe der Hauptkomponentenanalyse (PCA) transformiert wurden, um die Anonymität der Nutzer zu gewährleisten.

## Ziele
- **Entwicklung eines präzisen Klassifizierungsmodells**: Erarbeitung eines Modells, das effizient zwischen betrügerischen und legitimen Transaktionen unterscheiden kann.
- **Adressierung der Unausgeglichenheit im Datensatz**: Umsetzung von Techniken zur Bewältigung der starken Asymmetrie im Datensatz, insbesondere der geringen Anzahl an Betrugsfällen.
- **Evaluierung der Modellleistung**: Fokus auf Schlüsselmetriken wie Precision, Recall und F1-Score, um eine umfassende Bewertung der Modellgenauigkeit und -zuverlässigkeit sicherzustellen, insbesondere im Hinblick auf die Identifizierung von Betrugsfällen.

## Technologien_und_Werkzeuge
- **Programmiersprache**: Python
- **Hauptbibliotheken**: Pyspark
- **Datenvisualisierung**: Matplotlib,Databricks
- - **Entwicklungswerkzeuge**:Databricks community edition

## Daten
Der Datensatz enthält Transaktionen über zwei Tage mit 492 Betrugsfällen aus insgesamt 284,807 Transaktionen. Die Features 'Time' und 'Amount' sind die einzigen, die nicht durch PCA transformiert wurden.
Die Daten für dieses Projekt stammen von Kaggle, einer Online-Plattform für Datenwissenschaft und Maschinelles Lernen. Sie können den Originaldatensatz unter dem folgenden Link finden:
[Originaldatensatz auf Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

## Modellierung
Es werden verschiedene Klassifizierungsalgorithmen getestet und verglichen, einschließlich, aber nicht beschränkt auf:
- Random Forest
- Gradient Boosting
- LogisticRegression

## Ablauf

### 1. Analyse und transformieren der Quelldaten

Zu Beginn des Projekts konzentrierte ich mich auf die eingehende Analyse der vorhandenen Daten. Dabei stellte ich fest, dass die meisten Spalten der Daten mittels Principal Component Analysis (PCA) transformiert wurden. Dies beschränkte die unmittelbare Interpretierbarkeit der Daten, was eine Herausforderung darstellte. Zwei Merkmale jedoch, 'Time' und 'Amount', waren von dieser Transformation ausgenommen.

Für das Merkmal 'Amount' entschied ich mich für die Anwendung des RobustScaler, um die Daten zu normalisieren. Diese Methode wurde gewählt, da sie effizient mit Ausreißern umgeht, die in finanziellen Transaktionsdaten häufig vorkommen. Durch die Skalierung der 'Amount'-Daten konnte ich sicherstellen, dass sie in einem Bereich liegen, der für maschinelle Lernalgorithmen zugänglicher ist.

Beim Umgang mit dem Merkmal 'Time' ging ich anders vor. Hier skalierte ich die Werte in einen Bereich zwischen 0 und 1. Diese Normalisierung war entscheidend, um die zeitliche Dimension der Daten handhabbar zu machen und eine Überbetonung dieses Merkmals im Vergleich zu den anderen transformierten Merkmalen zu vermeiden.

| summary |       Scal_Amount       |   | summary |        Amount         |
|---------|-------------------------|---|---------|-----------------------|
| count   | 284807                  |   | count   | 284807                |
| mean    | 1.237562953507917       |   | mean    | 88.34961925093698     |
| stddev  | 3.503573459030513       |   | stddev  | 250.12010924018836    |
| min     | 0.0                     |   | min     | 0.0                   |
| max     | 359.87057010785827      |   | max     | 25691.16              |

| summary |       Time             |   | summary |       Scal_Time       |
|---------|-------------------------|---|---------|-----------------------|
| count   | 284807                  |   | count   | 284807                |
| mean    | 94813.85957508067       |   | mean    | 0.548716720537296     |
| stddev  | 47488.145954566266      |   | stddev  | 0.2748283829955438    |
| min     | 0.0                     |   | min     | 0.0                   |
| max     | 172792.0                |   | max     | 1.0                   |


### 2. Aufsetzen der ML Modelle
Nach der sorgfältigen Vorbereitung und Normalisierung der Daten habe ich mich darauf konzentriert, verschiedene Machine Learning-Modelle zu evaluieren, um das Potenzial jeder Methode in Bezug auf die spezifischen Anforderungen des Projekts zu bestimmen. Zu diesem Zweck habe ich folgende Modelle implementiert:

-**Logistic Regression**: Als klassisches Modell in der statistischen Modellierung und im maschinellen Lernen, wurde die logistische Regression eingesetzt, um eine Baseline für die Performance zu setzen und die Ergebnisse mit komplexeren Modellen zu vergleichen.
-**Random Forest**: Dieser Ansatz ist bekannt für seine Robustheit und gute Performance bei einer Vielzahl von Aufgaben. Random Forest eignet sich besonders gut für komplexe Datensätze, da er mehrere Entscheidungsbäume kombiniert und dadurch die Gefahr des Overfittings reduziert.
-**Gradient Boosting**: Als leistungsstarkes und flexibles Modell wurde Gradient Boosting aufgrund seiner Fähigkeit ausgewählt, schwache Vorhersagemodelle in ein starkes Gesamtmodell zu integrieren, was es besonders nützlich für ungleichmäßig verteilte Daten macht.
-**Linear Support Vector Classifier (SVC)**: Dieses Modell wurde aufgrund seiner Effektivität bei der Klassifizierung in hochdimensionalen Räumen ausgewählt, was es für die PCA-transformierten Daten besonders geeignet macht.

Bei der Analyse der Modellergebnisse habe ich mich besonders auf die Recall-Werte der als Betrug gekennzeichneten Klasse konzentriert. Da diese Klasse deutlich weniger Datenpunkte im Datensatz aufweist, ist der Recall-Wert von besonderer Bedeutung. Ein hoher Recall-Wert ist entscheidend, um sicherzustellen, dass die meisten Betrugsfälle korrekt identifiziert werden, selbst wenn das die Genauigkeit bei der Identifizierung legitimer Transaktionen leicht beeinträchtigt.

### 3 Downsampling
Nach der ersten Analyse und Bewertung der verschiedenen Modelle habe ich einen weiteren entscheidenden Schritt unternommen, um die Genauigkeit und Effizienz der Modelle zu verbessern. Mir wurde klar, dass das Ungleichgewicht zwischen der Anzahl der Betrugsfälle und der Anzahl der legitimen Transaktionen im Datensatz eine Herausforderung darstellte. Um dieses Problem zu adressieren und ein ausgewogeneres Trainingsset zu schaffen, habe ich die Anzahl der Nicht-Betrugsfälle auf dieselbe Anzahl wie die Betrugsfälle reduziert. Diese Technik, bekannt als Downsampling, hilft dabei, ein Modell zu trainieren, das nicht von der überwiegenden Klasse voreingenommen ist.

Mit diesem ausgewogeneren Datensatz habe ich die vier ausgewählten Modelle – Random Forest, Gradient Boosting, Logistic Regression und Linear Support Vector Classifier – erneut trainiert. Diese erneute Trainingssitzung ermöglichte es, die Leistungsfähigkeit der Modelle unter faireren und realistischeren Bedingungen zu beurteilen. Durch diese Anpassung erwartete ich eine signifikante Verbesserung im Recall-Wert für die Betrugserkennung, da das Modell nun gleichermaßen auf beide Klassen trainiert wurde.


Die Ergebnisse dieses zweiten Trainingsdurchlaufs waren aufschlussreich und zeigten deutliche Verbesserungen im Vergleich zu den anfänglichen Modellläufen.


##Ergbnis
