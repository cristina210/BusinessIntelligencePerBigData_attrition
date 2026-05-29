# Guida alle Run — Employee Attrition Classification
**Business Intelligence per Big Data — Politecnico di Torino, AA 2025-2026**

---

## Contesto generale

Il processo di analisi è stato condotto in modo **iterativo**: i risultati di classificazione ottenuti ad ogni run hanno influenzato le scelte di preprocessing, modificando il dataset di input per la run successiva. Il file `data_preprocessing.ipynb` va quindi inteso come il risultato finale di questo processo, non come un documento scritto in sequenza lineare. Il file `data_classification.ipynb` contiene i modelli finali addestrati sul dataset preprocessato risultante dall'ultima run.

I modelli usati come riferimento per valutare l'impatto delle scelte di preprocessing sono:
**Decision Tree (DT), Random Forest (RF), k-NN, Naïve Bayes (NB), SVM, MLP**

La metrica principale usata per confrontare le run è **F1-score sulla classe minoritaria (Attrition = 1)**, scelta per via dello sbilanciamento del dataset (~25% classe 1).

---

## Run 1 — Identificazione colonne da rimuovere per vincoli logici

### Obiettivo
Verificare se le colonne che violano vincoli di plausibilità temporale tra attributi (identificate nell'analisi wrong data) portino effettivo danno ai modelli di classificazione oppure se, pur essendo anomalie logiche, contengano ancora segnale predittivo utile.

### Contesto
Durante l'analisi dei wrong data (sezione 4.4 del preprocessing) sono state identificate violazioni di plausibilità temporale tra attributi:
- `TotalWorkingYears > Age`
- `YearsAtCompany > TotalWorkingYears`
- `YearsSinceLastPromotion > YearsAtCompany`
- `YearsWithCurrManager > YearsAtCompany`
- `YearsInCurrentRole > YearsAtCompany`

Le colonne coinvolte candidate alla rimozione erano: ['TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Age']

### Setup del dataset per questa run
- Missing values gestiti (Engagement_Score imputato con media del train)
- Random_Survey_Noise rimossa
- Outlier **non** rimossi
- Feature selection **non** ancora applicata
- Encoding e normalizzazione applicati

### Cosa si è testato
Si sono confrontate le performance dei 6 modelli nelle varie configurazioni ottenibili rimuovendo alcuni degli attributi in ['TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Age'] in modo da non violare i vincoli

### Risultato e decisione
La rimozione di `TotalWorkingYears` e `YearsAtCompany` non ha degradato le performance in modo generalizzato ed è risultata neutrale o leggermente positiva su alcuni modelli. Si è quindi deciso di **rimuovere entrambe le colonne** dal dataset.

**Flag nel codice:** `flag_rimozione_colonne_vincoli_logici = True`

---

## Run 2 — Valutazione dell'impatto della rimozione degli outlier

### Obiettivo
Verificare se la rimozione degli outlier identificati (tramite IQR univariato, distanza di Mahalanobis bivariata e DBSCAN multivariato) migliora le performance dei modelli oppure se questi valori estremi portino segnale predittivo rilevante.

### Contesto
L'analisi outlier (sezione 4.5 del preprocessing) aveva identificato:
- Outliers univariato (IQR)
- Outliers bivariati
- Outliers multivariati
La strategia conservativa adottata era di non rimuovere automaticamente, data la natura HR del dataset (valori estremi possono rappresentare profili reali di dipendenti e influenti sull'attrition).

### Setup del dataset per questa run
- Come Run 1, con `TotalWorkingYears` e `YearsAtCompany` già rimossi
- Si testano due varianti: dataset completo vs dataset con outlier rimossi (quelli confermati sia da DBSCAN che da almeno 2 analisi bivariate)


### Risultato e decisione
La rimozione degli outlier **non migliora in modo generalizzato** le performance:
- Naïve Bayes e Decision Tree restano sostanzialmente invariati
- Random Forest e MLP **peggiorano** sensibilmente (gli outlier portano segnale)
- Solo SVM beneficia della rimozione, coerentemente con la sua sensibilità ai valori estremi

Si è quindi deciso di **mantenere gli outlier nel dataset**.

---

## Run 3 — Feature selection: rimozione delle variabili poco informative

### Obiettivo
Identificare le variabili da rimuovere definitivamente dal dataset tra quelle candidate emerse nell'analisi esplorativa e nella feature selection (sezione 5.1 del preprocessing), verificando l'impatto sul F1.

### Contesto
Le analisi preliminari avevano individuato le seguenti **variabili candidate alla rimozione**:

| Variabile | Motivazione candidatura |
|---|---|
| `HourlyRate` | Correlazione con Attrition vicina a 0, nessun segnale discriminante |
| `DailyRate` | Stessa motivazione di HourlyRate |
| `Engagement_Score` | R²=0.77 su combinazione lineare di JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement → forte ridondanza; inoltre alto numero di missing values |
| `Tenure_Instability` | Ricostruibile con r=0.97 dal rapporto YearsSinceLastPromotion/YearsAtCompany → ridondanza quasi perfetta con feature già presenti |
| `OverTime` | Analisi visiva del tasso di attrition per categoria mostrava contributo limitato |
| `YearsSinceLastPromotion` | Valutata in combinazione con Tenure_Instability |

### Setup del dataset per questa run
- Come Run 2, outlier mantenuti.
- Si testano tutte le **combinazioni** (ablation study, sezione 10 del classification notebook) delle variabili candidate ['Engagement_Score', 'Tenure_Instability', 'HourlyRate', 'DailyRate', 'OverTime', 'YearsSinceLastPromotion' ], valutando F1 su tutti e 6 i modelli per ogni sottoinsieme

### Cosa si è testato
Ablation study esaustivo su `{Engagement_Score, Tenure_Instability, HourlyRate, DailyRate, OverTime, YearsSinceLastPromotion}`: tutte le 2⁶ = 64 combinazioni di rimozione testate con i parametri ottimali di ogni modello.

### Risultato e decisione
La configurazione ottimale risultante dall'ablation study ha portato alla rimozione di:
- `HourlyRate` — nessun contributo, rimozione neutrale o positiva su tutti i modelli.

Le variabili `Engagement_Score`, `Tenure_Instability`, `OverTime` e `YearsSinceLastPromotion` sono state **mantenute** nonostante la ridondanza o il segnale debole, perché la loro rimozione degradava le performance su almeno un sottoinsieme di modelli e perché portano interpretabilità di dominio rilevante per l'analisi HR.

**Lista finale colonne rimosse** (variabile `cols_to_remove_fs` nel codice):
`['HourlyRate']`

---

## Dataset finale

Il dataset risultante dalle tre run e usato nel notebook di classificazione è:

| Versione | File | Contenuto |
|---|---|---|
| Continua | `train_with_label_dataPrep.xlsx` | Feature numeriche originali, encoding one-hot nominali, MinMaxScaler |
| Discretizzata | `train_with_label_dataPrep_discr.xlsx` | Stessa pipeline + discretizzazione Age, MonthlyIncome, DistanceFromHome, Tenure_Instability, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager |
| Test continua | `test_dataPrep.xlsx` | Stessa trasformazione del train (scaler e bin fittati solo su train) |
| Test discretizzata | `test_dataPrep_discr.xlsx` | Idem |

### Colonne rimosse in totale rispetto al dataset originale

| Colonna | Motivo | Run |
|---|---|---|
| `Random_Survey_Noise` | Rumore esplicito, non feature informativa | Pre-run |
| `TotalWorkingYears` | Vincoli logici + ridondanza con altre variabili temporali | Run 1 |
| `YearsAtCompany` | Vincoli logici + ridondanza con altre variabili temporali | Run 1 |
| `HourlyRate` | Nessun contributo predittivo (ablation study) | Run 3 |
