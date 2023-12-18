[df_result_down.csv](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/files/13702269/df_result_down.csv)[df_result.csv](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/files/13702266/df_result.csv)# Kreditkarten-Betrugserkennung

## Inhaltsverzeichnis
  - [Projektübersicht](#Projektübersicht)
  - [Ziele](#Ziele)
  - [Technologien und Werkzeuge](#Technologien_und_Werkzeuge)
  - [Daten](#Daten)
  - [Modellierung](#Modellierung)
  - [Ablauf](#Ablauf)
    - [1. Analyse und transformieren der Quelldaten](#1._Analyse_und_transformieren_der_Quelldaten)
    - [2. Aufsetzen der ML Modelle](#2._Aufsetzen_der_ML_Modelle)
    - [3. Downsampling](#3._Downsampling)
  - [Ergbnis](#Ergbnis)
  - [Weiter Schritte](#Weiter_Schritte)

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

Zu Beginn des Projekts konzentrierte ich mich auf die eingehende Analyse der vorhandenen Daten. Dabei stellte ich fest, dass die meisten Spalten der Daten mittels Principal Component Analysis (PCA) transformiert wurden. Dies beschränkte die unmittelbare Interpretierbarkeit der Daten, was eine Herausforderung darstellte. Zwei Merkmale jedoch,'Class', 'Time' und 'Amount', waren von dieser Transformation ausgenommen.

![hist_all_with-back](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/468a664c-fdf3-410d-b6b4-699a37402e8b)


Für das Merkmal 'Amount' entschied ich mich für die Anwendung des RobustScaler, um die Daten zu normalisieren. Diese Methode wurde gewählt, da sie effizient mit Ausreißern umgeht, die in finanziellen Transaktionsdaten häufig vorkommen. Durch die Skalierung der 'Amount'-Daten konnte ich sicherstellen, dass sie in einem Bereich liegen, der für maschinelle Lernalgorithmen zugänglicher ist.

Beim Umgang mit dem Merkmal 'Time' ging ich anders vor. Hier skalierte ich die Werte in einen Bereich zwischen 0 und 1. Diese Normalisierung war entscheidend, um die zeitliche Dimension der Daten handhabbar zu machen und eine Überbetonung dieses Merkmals im Vergleich zu den anderen transformierten Merkmalen zu vermeiden.

Das Ergbnis des Transformieren:

| summary |        Amount         |   | summary |       Scal_Amount       |
|---------|-----------------------|---|---------|-------------------------|
| count   | 284807                |   | count   | 284807                  |
| mean    | 88.34961925093698     |   | mean    | 1.237562953507917       |
| stddev  | 250.12010924018836    |   | stddev  | 3.503573459030513       |
| min     | 0.0                   |   | min     | 0.0                     |
| max     | 25691.16              |   | max     | 359.87057010785827      |

| summary |       Time            |   | summary |       Scal_Time         |
|---------|-----------------------|---|---------|-------------------------|
| count   | 284807                |   | count   | 284807                  |
| mean    | 94813.85957508067     |   | mean    | 0.548716720537296       |
| stddev  | 47488.145954566266    |   | stddev  | 0.2748283829955438      |
| min     | 0.0                   |   | min     | 0.0                     |
| max     | 172792.0              |   | max     | 1.0                     |



### 2. Aufsetzen der ML Modelle
Nach der sorgfältigen Vorbereitung und Normalisierung der Daten habe ich mich darauf konzentriert, verschiedene Machine Learning-Modelle zu evaluieren, um das Potenzial jeder Methode in Bezug auf die spezifischen Anforderungen des Projekts zu bestimmen. Zu diesem Zweck habe ich folgende Modelle implementiert:

-**Logistic Regression**: Als klassisches Modell in der statistischen Modellierung und im maschinellen Lernen, wurde die logistische Regression eingesetzt, um eine Baseline für die Performance zu setzen und die Ergebnisse mit komplexeren Modellen zu vergleichen.
-**Random Forest**: Dieser Ansatz ist bekannt für seine Robustheit und gute Performance bei einer Vielzahl von Aufgaben. Random Forest eignet sich besonders gut für komplexe Datensätze, da er mehrere Entscheidungsbäume kombiniert und dadurch die Gefahr des Overfittings reduziert.
-**Gradient Boosting**: Als leistungsstarkes und flexibles Modell wurde Gradient Boosting aufgrund seiner Fähigkeit ausgewählt, schwache Vorhersagemodelle in ein starkes Gesamtmodell zu integrieren, was es besonders nützlich für ungleichmäßig verteilte Daten macht.
-**Linear Support Vector Classifier (SVC)**: Dieses Modell wurde aufgrund seiner Effektivität bei der Klassifizierung in hochdimensionalen Räumen ausgewählt, was es für die PCA-transformierten Daten besonders geeignet macht.

Bei der Analyse der Modellergebnisse habe ich mich besonders auf die Recall-Werte der als Betrug gekennzeichneten Klasse konzentriert. Da diese Klasse deutlich weniger Datenpunkte im Datensatz aufweist, ist der Recall-Wert von besonderer Bedeutung. Ein hoher Recall-Wert ist entscheidend, um sicherzustellen, dass die meisten Betrugsfälle korrekt identifiziert werden, selbst wenn das die Genauigkeit bei der Identifizierung legitimer Transaktionen leicht beeinträchtigt.

![plot_df_result](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/d128efa9-9592-4717-a15c-8e272fba392d)

[UploadiF1 Score,Label,Model,Precision,Recall
0.9996820574416221,0,lr,0.9994349085258176,0.9999293286219081
0.7804878048780487,1,lr,0.9411764705882353,0.6666666666666666
0.9997880083383387,0,gbt,0.9996820462092842,0.9998939929328622
0.8666666666666666,1,gbt,0.9285714285714286,0.8125
0.9996997262209664,0,lsvc,0.9994349284831362,0.9999646643109541
0.7901234567901234,1,lsvc,0.9696969696969697,0.6666666666666666
0.9997880233174351,0,rf,0.9996114447191805,0.9999646643109541
0.8604651162790697,1,rf,0.9736842105263158,0.7708333333333334
ng df_result.csv…]()


### 3. Downsampling
Nach der ersten Analyse und Bewertung der verschiedenen Modelle habe ich einen weiteren entscheidenden Schritt unternommen, um die Genauigkeit und Effizienz der Modelle zu verbessern. Mir wurde klar, dass das Ungleichgewicht zwischen der Anzahl der Betrugsfälle und der Anzahl der legitimen Transaktionen im Datensatz eine Herausforderung darstellte. Um dieses Problem zu adressieren und ein ausgewogeneres Trainingsset zu schaffen, habe ich die Anzahl der Nicht-Betrugsfälle auf dieselbe Anzahl wie die Betrugsfälle reduziert. Diese Technik, bekannt als Downsampling, hilft dabei, ein Modell zu trainieren, das nicht von der überwiegenden Klasse voreingenommen ist.

Mit diesem ausgewogeneren Datensatz habe ich die vier ausgewählten Modelle – Random Forest, Gradient Boosting, Logistic Regression und Linear Support Vector Classifier – erneut trainiert. Diese erneute Trainingssitzung ermöglichte es, die Leistungsfähigkeit der Modelle unter faireren und realistischeren Bedingungen zu beurteilen. Durch diese Anpassung erwartete ich eine signifikante Verbesserung im Recall-Wert für die Betrugserkennung, da das Modell nun gleichermaßen auf beide Klassen trainiert wurde.

Die Ergebnisse dieses zweiten Trainingsdurchlaufs waren aufschlussreich und zeigten deutliche Verbesserungen im Vergleich zu den anfänglichen Modellläufen.

![plot_df_result_down](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/b8c4ac50-dfb6-4236-b1bc-34860613a2b5)

[Uploading df_rF1 Score,Label,Model,Precision,Recall
0.8971962616822429,0,lr,0.8421052631578947,0.96
0.8971962616822429,1,lr,0.96,0.8421052631578947
0.9259259259259259,0,rf,0.8620689655172413,1
0.9245283018867925,1,rf,1,0.8596491228070176
0.8431372549019608,0,gbt,0.8269230769230769,0.86
0.8571428571428571,1,gbt,0.8727272727272727,0.8421052631578947
0.882882882882883,0,lsvc,0.8032786885245902,0.98
0.8737864077669902,1,lsvc,0.9782608695652174,0.7894736842105263
esult_down.csv…]()

## Ergbnis

## Weiter Schritte
