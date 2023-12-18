# Kreditkarten-Betrugserkennung
![Uploading DALL·E 2023-12-18 12.32.51 - A banner depicting the theme of credit card fraud. The image features a large, ominous-looking credit card casting a shadow over a background of scatt.png…]()


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
  - [Ergebnis](#Ergebnis)
  - [Weiter Schritte](#Weiter_Schritte)
  - [Data Bricks Notebook](###Link_zum_Databricks_Notebook)

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

Das Ergebnis des Transformieren:

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
[Ergebnis als CSV](Data/train_result.csv)


### 3. Downsampling
Nach der ersten Analyse und Bewertung der verschiedenen Modelle habe ich einen weiteren entscheidenden Schritt unternommen, um die Genauigkeit und Effizienz der Modelle zu verbessern. Mir wurde klar, dass das Ungleichgewicht zwischen der Anzahl der Betrugsfälle und der Anzahl der legitimen Transaktionen im Datensatz eine Herausforderung darstellte. Um dieses Problem zu adressieren und ein ausgewogeneres Trainingsset zu schaffen, habe ich die Anzahl der Nicht-Betrugsfälle auf dieselbe Anzahl wie die Betrugsfälle reduziert. Diese Technik, bekannt als Downsampling, hilft dabei, ein Modell zu trainieren, das nicht von der überwiegenden Klasse voreingenommen ist.

Mit diesem ausgewogeneren Datensatz habe ich die vier ausgewählten Modelle – Random Forest, Gradient Boosting, Logistic Regression und Linear Support Vector Classifier – erneut trainiert. Diese erneute Trainingssitzung ermöglichte es, die Leistungsfähigkeit der Modelle unter faireren und realistischeren Bedingungen zu beurteilen. Durch diese Anpassung erwartete ich eine signifikante Verbesserung im Recall-Wert für die Betrugserkennung, da das Modell nun gleichermaßen auf beide Klassen trainiert wurde.

Die Ergebnisse dieses zweiten Trainingsdurchlaufs waren aufschlussreich und zeigten deutliche Verbesserungen im Vergleich zu den anfänglichen Modellläufen.

![plot_df_result_down](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/b8c4ac50-dfb6-4236-b1bc-34860613a2b5)
[Ergebnis als CSV](Data/train_result_down.csv)


## Ergebnis
Die Bewertung der Modellleistung von dem Gradient Boosting Model zeigt, dass das Downsampling des Datensatzes, bei dem die Mehrheitsklasse reduziert wird, um ein ausgeglicheneres Verhältnis zwischen den Klassen zu erreichen, zu unterschiedlichen Ergebnissen führt. Das Modell erzielt für die Klasse 0.0 nach dem Downsampling eine Präzision von nahezu 1,0 und einen Recall von etwa 0,944, was auf eine effektive Erkennung der tatsächlichen Fälle der Mehrheitsklasse hinweist, mit einer hohen Vermeidungsrate von Falsch-Positiven. Der F1 Score von rund 0,971 bestätigt diese hohe Leistung.

Für die Klasse 1.0, die potenziell betrügerische Überweisungen darstellt, zeigt das Modell nach dem Downsampling einen extrem hohen Recall von 0,975, was bedeutet, dass fast alle tatsächlichen Betrugsfälle identifiziert werden. Allerdings ist die Präzision mit einem Wert von 0,024 sehr gering, was auf eine hohe Anzahl von Falsch-Positiven hinweist; die meisten der als Betrug erkannten Fälle sind tatsächlich keine. Dies resultiert in einem niedrigen F1 Score von ungefähr 0,047, was auf eine suboptimale Balance zwischen Precision und Recall hindeutet.

Ohne Downsampling weist das Modell für die Klasse 0.0 nahezu perfekte Präzision und Recall auf, was sich in einem F1 Score von ungefähr 0,999 widerspiegelt, ein Indikator für eine nahezu fehlerfreie Klassifikation. Bei der Klasse 1.0 ist die Präzision mit einem Wert von rund 0,941 deutlich höher als beim downgesampelten Modell, während der Recall auf 0,8 sinkt. Der daraus resultierende F1 Score von etwa 0,865 zeigt eine wesentlich bessere Balance zwischen Precision und Recall als beim downgesampelten Modell.

Diese Ergebnisse verdeutlichen, wie entscheidend die Wahl der Datenvorbereitungsmethoden für die Modellleistung ist. Während das Downsampling dazu beiträgt, die Erkennung seltener Ereignisse zu maximieren, kann es gleichzeitig die Anzahl der Falsch-Positiven erhöhen, was in der Praxis zu zusätzlichem Aufwand bei der Überprüfung der als betrügerisch gekennzeichneten Transaktionen führen kann. Umgekehrt kann ein unausgeglichener Datensatz ohne Downsampling zu einer höheren Genauigkeit bei der Identifizierung von Betrugsfällen führen, jedoch auf Kosten einer potenziell niedrigeren Entdeckungsrate. Dies betont die Notwendigkeit, eine ausgewogene Strategie zu finden, die sowohl hohe Präzision als auch hohen Recall bietet, um die Effizienz und Wirksamkeit in der Betrugserkennung zu optimieren.

Auf Basis dieser Beobachtungen und der Tatsache, dass das Gradient Boosting Modell die besten Werte auf den Validierungsdaten lieferte, wurde dieses Modell ausgewählt, um auf dem Testdatensatz angewendet zu werden.

Das Ergbnis mit den Testdaten Ergibt:

Gradient Boosting ohne Downsampling:

| Label | Precision          | Recall             | F1 Score          |
|-------|--------------------|--------------------|-------------------|
| 0.0   | 0.9997161711487973 | 0.9999290276792051 | 0.9998225880850158|
| 1.0   | 0.9411764705882353 | 0.8                | 0.8648648648648648|

Gradient Boosting mit Downsampling:

| Label | Precision              | Recall             | F1 Score          |
|-------|------------------------|--------------------|-------------------|
| 0.0   | 0.9999624173180999     | 0.944180269694819  | 0.9712710812586699|
| 1.0   | 0.024193548387096774   | 0.975              | 0.0472154963680387|

## Weiter Schritte
Um die Leistung unseres Klassifikationsmodells zu verbessern und es weiter zu optimieren, sollten wir uns auf Hyperparameter-Tuning konzentrieren:

Zunächst ist es wichtig, das Hyperparameter-Tuning systematisch anzugehen. Dafür können wir Techniken wie Grid Search oder Random Search nutzen. Diese Methoden ermöglichen es, verschiedene Kombinationen von Hyperparametern systematisch zu testen.
Konkret sollten wir uns auf Schlüsselparameter wie die Lernrate, die Anzahl der Bäume in Entscheidungsbaum-basierten Modellen oder die Anzahl der Schichten und Neuronen in neuronalen Netzen konzentrieren.
Der Plan wäre, mit einem breiten Spektrum an Parametereinstellungen zu beginnen und diese allmählich zu verfeinern, um die Kombination zu finden, die die beste Leistung zeigt.

## Link zum Databricks Notebook
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1881572392991630/2975787479060580/5681538385804398/latest.html
