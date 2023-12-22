# Kreditkarten-Betrugserkennung
![DALL·E 2023-12-18 12 35 06 - A banner depicting the theme of credit card fraud, sized 1800x600 pixels  The image showcases a large, threatening credit card overshadowing a backgro](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/4fc5bdeb-2189-40fa-a2b5-a0bf87a69145)


## Inhaltsverzeichnis
  - [Projektübersicht](pProjektübersicht)
  - [Ziele](#ziele)
  - [Technologien und Werkzeuge](#technologien-und-werkzeuge)
  - [Daten](#daten)
  - [Modellierung](#modellierung)
  - [Ablauf](#ablauf)
    - [1. Analyse und transformieren der Quelldaten](#analyse-und-transformieren-der-quelldaten)
    - [2. Aufsetzen der ML Modelle](##aufsetzen-der-ml-modelle)
    - [3. Downsampling](#downsampling)
  - [Ergebnis](#ergebnis)
  - [Weiter Schritte](#weiter-schritte)
  - [Data Bricks Notebook](#link-zum-databricks-notebook)
  - [Credits](#credits)

## Projektübersicht
Dieses Projekt zielt darauf ab, betrügerische Transaktionen in Kreditkartendaten unter Verwendung fortschrittlicher Machine Learning-Techniken zu identifizieren. Der verwendete Datensatz besteht aus Transaktionen von europäischen Karteninhabern im September 2013, die mithilfe der Hauptkomponentenanalyse (PCA) transformiert wurden, um die Anonymität der Nutzer zu gewährleisten.

## Ziele
- **Entwicklung eines präzisen Klassifizierungsmodells**: Erarbeitung eines Modells, das effizient zwischen betrügerischen und legitimen Transaktionen unterscheiden kann.
- **Adressierung der Unausgeglichenheit im Datensatz**: Umsetzung von Techniken zur Bewältigung der starken Asymmetrie im Datensatz, insbesondere der geringen Anzahl an Betrugsfällen.
- **Evaluierung der Modellleistung**: Fokus auf Schlüsselmetriken wie Precision, Recall und F1-Score, um eine umfassende Bewertung der Modellgenauigkeit und -zuverlässigkeit sicherzustellen, insbesondere im Hinblick auf die Identifizierung von Betrugsfällen.

## Technologien undWerkzeuge
- **Programmiersprache**: Python
- **Hauptbibliotheken**: Pyspark
- **Datenvisualisierung**: Matplotlib,Databricks
- **Entwicklungswerkzeuge**:Databricks community edition

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

1. ## Analyse und transformieren der Quelldaten

Zu Beginn des Projekts widmete ich mich einer gründlichen Analyse der vorliegenden Daten. Dabei bemerkte ich, dass die Mehrheit der Datenfelder transformiert worden war, was ihre unmittelbare Interpretierbarkeit einschränkte und eine Herausforderung darstellte. Die Attribute 'Class', 'Time' und 'Amount' waren jedoch von dieser Transformation ausgenommen.

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



2. ### Aufsetzen der ML Modelle
Nach der sorgfältigen Vorbereitung und Normalisierung der Daten, wobei ich darauf geachtet habe, die Daten in 80% Trainings-, 10% Test- und 10% Validierungsdaten aufzuteilen, habe ich mich darauf konzentriert, verschiedene Machine Learning-Modelle zu evaluieren. Ziel war es, das Potenzial jeder Methode in Bezug auf die spezifischen Anforderungen des Projekts zu bestimmen. Zu diesem Zweck habe ich folgende Modelle implementiert:

- **Logistic Regression**: Als klassisches Modell in der statistischen Modellierung und im maschinellen Lernen, wurde die logistische Regression eingesetzt, um eine Baseline für die Performance zu setzen und die Ergebnisse mit komplexeren Modellen zu vergleichen.
- **Random Forest**: Dieser Ansatz ist bekannt für seine Robustheit und gute Performance bei einer Vielzahl von Aufgaben. Random Forest eignet sich besonders gut für komplexe Datensätze, da er mehrere Entscheidungsbäume kombiniert und dadurch die Gefahr des Overfittings reduziert.
- **Gradient Boosting**: Als leistungsstarkes und flexibles Modell wurde Gradient Boosting aufgrund seiner Fähigkeit ausgewählt, schwache Vorhersagemodelle in ein starkes Gesamtmodell zu integrieren, was es besonders nützlich für ungleichmäßig verteilte Daten macht.
- **Linear Support Vector Classifier (SVC)**: Dieses Modell wurde aufgrund seiner Effektivität bei der Klassifizierung in hochdimensionalen Räumen ausgewählt, was es für die PCA-transformierten Daten besonders geeignet macht.

Bei der Analyse der Modellergebnisse habe ich mich besonders auf die Recall-Werte der als Betrug gekennzeichneten Klasse konzentriert. Da diese Klasse deutlich weniger Datenpunkte im Datensatz aufweist, ist der Recall-Wert von besonderer Bedeutung. Ein hoher Recall-Wert ist entscheidend, um sicherzustellen, dass die meisten Betrugsfälle korrekt identifiziert werden, selbst wenn das die Genauigkeit bei der Identifizierung legitimer Transaktionen leicht beeinträchtigt.

![plot_df_result](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/d128efa9-9592-4717-a15c-8e272fba392d)
[Ergebnis als CSV](Data/train_result.csv)


3. ### Downsampling
Nach der ersten Analyse und Bewertung der verschiedenen Modelle habe ich einen weiteren entscheidenden Schritt unternommen, um die Genauigkeit und Effizienz der Modelle zu verbessern. Mir wurde klar, dass das Ungleichgewicht zwischen der Anzahl der Betrugsfälle und der Anzahl der legitimen Transaktionen im Datensatz eine Herausforderung darstellte. Um dieses Problem zu adressieren und ein ausgewogeneres Trainingsset zu schaffen, habe ich die Anzahl der Nicht-Betrugsfälle auf dieselbe Anzahl wie die Betrugsfälle reduziert. Diese Technik, bekannt als Downsampling, hilft dabei, ein Modell zu trainieren, das nicht von der überwiegenden Klasse voreingenommen ist.

Mit diesem ausgewogeneren Datensatz habe ich die vier ausgewählten Modelle – Random Forest, Gradient Boosting, Logistic Regression und Linear Support Vector Classifier – erneut trainiert. Diese erneute Trainingssitzung ermöglichte es, die Leistungsfähigkeit der Modelle unter faireren und realistischeren Bedingungen zu beurteilen. Durch diese Anpassung erwartete ich eine signifikante Verbesserung im Recall-Wert für die Betrugserkennung, da das Modell nun gleichermaßen auf beide Klassen trainiert wurde.

Die Ergebnisse dieses zweiten Trainingsdurchlaufs waren aufschlussreich und zeigten deutliche Verbesserungen im Vergleich zu den anfänglichen Modellläufen.

![newplot-2](https://github.com/Murgatroyd-K/Kreditkarten_Betrugserkennung/assets/76660447/b0df8b1c-47d6-49e0-ab4b-a31e46b0d6ac)
[Ergebnis als CSV](Data/train_result_down.csv)


## Ergebnis
Die Bewertung der Modellleistung des Gradient Boosting Modells zeigt, dass das Downsampling des Datensatzes, bei dem die Mehrheitsklasse reduziert wird, zu unterschiedlichen Ergebnissen führt. Nach dem Downsampling erreicht das Modell für die Klasse 'Kein Betrug' eine Präzision von nahezu 0,9998876656931027 und einen Recall von etwa 0,9475869410929737. Dies deutet auf eine effektive Erkennung der tatsächlichen Fälle der Mehrheitsklasse hin, mit einer hohen Vermeidungsrate von Falsch-Positiven. Der F1 Score von rund 0,9730350180373866 bestätigt diese hohe Leistung.

Für die Klasse 'Betrug', die potenziell betrügerische Überweisungen repräsentiert, zeigt das Modell nach dem Downsampling einen hohen Recall von 0,925. Dies bedeutet, dass fast alle tatsächlichen Betrugsfälle identifiziert werden, jedoch ist die Präzision mit einem Wert von 0,02443857331571995 sehr gering. Diese niedrige Präzision resultiert in einer hohen Anzahl von Falsch-Positiven, was bedeutet, dass viele der als Betrug erkannten Fälle tatsächlich legitim sind. Daraus ergibt sich ein F1 Score von ungefähr 0,047619047619047616, was eine suboptimale Balance zwischen Precision und Recall anzeigt.

Diese Ergebnisse spiegeln die Komplexität und Herausforderungen wider, die mit dem Downsampling in der Praxis verbunden sind. Es zeigt sich, dass, obwohl das Downsampling effektiv dabei sein kann, seltene Ereignisse wie Betrug zu identifizieren, es gleichzeitig zu einer erhöhten Anzahl von falsch positiven Ergebnissen führen kann. Dies kann in der Praxis zu zusätzlichem Aufwand bei der Überprüfung der als betrügerisch gekennzeichneten Transaktionen führen.

Im Gegensatz dazu zeigt das Modell ohne Downsampling für die Klasse 'Kein Betrug' nahezu perfekte Präzision und Recall, was sich in einem F1 Score von etwa 0,999 widerspiegelt. Dies ist ein Indikator für eine nahezu fehlerfreie Klassifikation. Bei der Klasse 'Betrug' ist die Präzision mit einem Wert von rund 0,941 deutlich höher als beim downgesampelten Modell, während der Recall auf 0,8 sinkt. Der daraus resultierende F1 Score von etwa 0,865 zeigt eine wesentlich bessere Balance zwischen Precision und Recall als beim downgesampelten Modell.

Diese Beobachtungen unterstreichen die Notwendigkeit einer ausgewogenen Strategie bei der Datenvorbereitung und Modellbildung. Es ist entscheidend, eine Balance zu finden, die sowohl hohe Präzision als auch hohen Recall bietet, um die Effizienz und Wirksamkeit in der Betrugserkennung zu optimieren. Basierend auf diesen Überlegungen und der Tatsache, dass das Gradient Boosting Modell die besten Ergebnisse auf den Validierungsdaten zeigte, wurde es ausgewählt, um auf dem Testdatensatz angewendet zu werden.

Gradient Boosting ohne Downsampling:

| Label | Precision          | Recall             | F1 Score          |
|-------|--------------------|--------------------|-------------------|
| 0.0   | 0.9997161711487973 | 0.9999290276792051 | 0.9998225880850158|
| 1.0   | 0.9411764705882353 | 0.8                | 0.8648648648648648|

Gradient Boosting mit Downsampling:

| Label | Precision              | Recall             | F1 Score          |
|-------|------------------------|--------------------|-------------------|
| 0.0   | 0.9998876656931027     | 0.9475869410929737 | 0.9730350180373866|
| 1.0   | 0.02443857331571995    | 0.925              | 0.0476190476190476|

## Weiter Schritte
Um die Leistung des Klassifikationsmodells zu verbessern und es weiter zu optimieren, sollte man sich auf Hyperparameter-Tuning konzentrieren:

Zunächst ist es wichtig, das Hyperparameter-Tuning systematisch anzugehen. Dafür können man Techniken wie Grid Search oder Random Search nutzen. Diese Methoden ermöglichen es, verschiedene Kombinationen von Hyperparametern systematisch zu testen.
Konkret sollten man sich auf Schlüsselparameter wie die Lernrate oder die Anzahl der Bäume in Entscheidungsbaum-basierten Modellen konzentrieren.
Der Plan wäre, mit einem breiten Spektrum an Parametereinstellungen zu beginnen und diese allmählich zu verfeinern, um die Kombination zu finden, die die beste Leistung zeigt.

## Link zum Databricks Notebook
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1881572392991630/2975787479060580/5681538385804398/latest.html
## Credits
Text und Titelbild: Generiert von GPT-4 (OpenAI)
Daten: [Originaldatensatz auf Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
