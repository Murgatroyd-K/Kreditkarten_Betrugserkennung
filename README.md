# Kreditkarten-Betrugserkennung

## Projektübersicht
Dieses Projekt zielt darauf ab, betrügerische Transaktionen in Kreditkartendaten unter Verwendung fortschrittlicher Machine Learning-Techniken zu identifizieren. Der verwendete Datensatz besteht aus Transaktionen von europäischen Karteninhabern im September 2013, die mithilfe der Hauptkomponentenanalyse (PCA) transformiert wurden, um die Anonymität der Nutzer zu gewährleisten.

## Ziele
- **Entwicklung eines präzisen Klassifizierungsmodells**: Erstellung eines Modells, das in der Lage ist, zwischen betrügerischen und legitimen Transaktionen effektiv zu unterscheiden.
- **Adressierung der Unausgeglichenheit im Datensatz**: Implementierung von Strategien, um mit der starken Unausgeglichenheit im Datensatz umzugehen, wobei betrügerische Transaktionen nur einen kleinen Teil der Daten ausmachen.
- **Evaluierung der Modellleistung**: Verwendung geeigneter Metriken wie der Area Under the Precision-Recall Curve (AUPRC), um die Leistung des Modells zu beurteilen.

## Technologien und Werkzeuge
- **Programmiersprache**: Python
- **Hauptbibliotheken**: Pyspark
- **Datenvisualisierung**: Matplotlib

## Daten
Der Datensatz enthält Transaktionen über zwei Tage mit 492 Betrugsfällen aus insgesamt 284,807 Transaktionen. Die Features 'Time' und 'Amount' sind die einzigen, die nicht durch PCA transformiert wurden.

## Modellierung
Es werden verschiedene Klassifizierungsalgorithmen getestet und verglichen, einschließlich, aber nicht beschränkt auf:
- Random Forest
- Gradient Boosting

## Datenquelle
Die Daten für dieses Projekt stammen von Kaggle, einer Online-Plattform für Datenwissenschaft und Maschinelles Lernen. Sie können den Originaldatensatz unter dem folgenden Link finden:

[Originaldatensatz auf Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
