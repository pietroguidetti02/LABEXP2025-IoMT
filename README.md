# LABEXP2025-IoMT
repo for lab exp i've done at polimi in 2025


# 🩺 6-Minute Walk Test Dashboard – Indoor Wearable Tracking System

Questo progetto consente di tracciare in tempo reale la distanza percorsa da un paziente indossando un wearable BLE e visualizzarla su una dashboard interattiva. È pensato per essere usato in ambienti ospedalieri indoor come test dei 6 minuti (6MWT).

## 📌 Requisiti

- Sistema operativo: **Windows**
- IDE consigliato: **PyCharm**
- Python ≥ 3.8
- Librerie necessarie:
  - `dash`
  - `plotly`
  - `numpy`
  - `pandas`
  - `json`
  - `socket`
  - `threading`
  - (facoltativo: `subprocess`, `os`, `datetime`, ecc.)

Puoi installare le dipendenze con:

```bash
pip install dash plotly pandas numpy
```

---

## 🚀 Come avviare il progetto

### 1. **Avvio del backend (server ricezione dati BLE)**
Apri PyCharm e avvia lo script:
```
enhanced-distance-tracking_v3_jsonPatient.py
```

Questo script:
- Attiva un server TCP per ricevere dati da più ancore BLE.
- Crea una cartella di lavoro (`positions_data_*`) dove salva:
  - `info.json` con i dati del paziente.
  - `wearable_data.csv` con tutti i dati raccolti (RSSI, distanza stimata, passi, posizione, ecc.).

### 2. **Avvio della dashboard (frontend)**
Sempre in PyCharm, avvia lo script:
```
v7_wearable_dashboard_visualization-program_multiPatient.py
```

Dopo qualche secondo, la dashboard sarà disponibile all’indirizzo:

📍 [http://localhost:8050](http://localhost:8050)

---

## 🧪 Procedura per iniziare un test

1. Inserisci i **dati del paziente** nei campi in alto (nome, età, altezza, genere).
2. Clicca su **"Confirm Patient Data"**:
   - Verrà creato un file `info.json` con i dati del paziente nella nuova cartella.
   - Il sistema è pronto per iniziare a ricevere dati.

3. Clicca su **"Start New 6-Minute Walk Test"**:
   - Il sistema inizierà a scrivere i dati nel CSV ogni volta che riceve un pacchetto BLE valido da una o più ancore.
   - ⚠️ Potrebbe esserci un piccolo **ritardo (latenza)** nella visualizzazione dei dati, dovuto sia all’hardware (CPU) sia alla scrittura/lettura dal file.

4. Clicca su **"Start Timer"**:
   - Inizia la sessione ufficiale da **6 minuti**.
   - Il timer viene mostrato in alto con indicazione del tempo trascorso e residuo.

---

## 📊 Funzionalità della dashboard

- Visualizzazione della **posizione stimata** del wearable in tempo reale.
- Tracciamento delle **distanze** stimate con:
  - Calcolo dai passi (pedometria)
  - Calcolo dalla posizione (trilaterazione BLE)
  - Distanza "fusa" (fusione delle due precedenti).
- Grafici di **RSSI**, **qualità del segnale**, **heatmap**, **stabilità della posizione**, **velocità** e **attività** (in movimento o fermo).

---

## ⚠️ Note

- La frequenza di aggiornamento dipende dalla velocità del tuo PC e dalla frequenza di invio delle ancore BLE.
- I dati vengono aggiornati **ogni 0.5 secondi**, ma è normale vedere una latenza di qualche secondo nella visualizzazione.
- Le prestazioni possono essere migliorate ottimizzando il codice o usando hardware più potente.

---

## 🧹 Pulizia dei dati

I dati salvati vengono archiviati in cartelle `positions_data_<id>_<timestamp>`. Puoi fare il backup o cancellarli a mano dopo ogni test.

---

## 🧑‍💻 Autori

- Pietro Guidetti  
- Minglei Shao  
- Alice, Ameni, IO  
*IoMT 2025 | LABEXP – Indoor Health Monitoring*

---
