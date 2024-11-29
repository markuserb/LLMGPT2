Voraussetzungen

	Bevor Sie starten, stellen Sie sicher, dass folgende Software und Pakete installiert sind:

	•	Python 3.8 oder neuer
	•	Abhängigkeiten aus requirements.txt installieren: pip install -r requirements.txt


Daten vorbereiten

	1. 	Trainingsdaten bereitstellen:
	Speichern Sie Ihre Trainingsdaten in einem geeigneten Format (z. B. .txt).
	Legen Sie die Dateien im Verzeichnis data/ ab.


Modell trainieren

	1.	Training starten:
	Verwenden Sie das Skript train.py, um das Modell direkt zu trainieren.
	2.	Zwischenspeichern und Ergebnisse:
	Das Skript speichert die trainierten Modelle und Checkpoints im Verzeichnis results/.


Modell evaluieren

	1.	Evaluation starten:
	Nachdem das Modell trainiert wurde, können Sie es auf einem Testdatensatz evaluieren, um seine Leistung zu überprüfen. Führen Sie dazu das Skript evaluate.py aus

 	2.	Auswertungsergebnisse:
	Das Skript gibt die Metriken wie Perplexity oder andere Evaluierungskennzahlen aus, die zur Beurteilung der Modellgüte verwendet werden.


Modell testen

	Nach der Evaluation können Sie das Modell auch weiterhin mit dem Skript generate.py testen



