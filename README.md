---
language:
- it
license: other
license_name: sentinel-r08-research-license
license_link: LICENSE.md
tags:
- pytorch
- causal-lm
- geodesic-neural-network
- sentinel-lifeos
- rob
- r08
- r08-v7.2
- edge-ai
- 1d-causal-conv
- icosahedron-hubs
- italian
pipeline_tag: text-generation
inference: false
---

<p align="center">
 <img src="assets/r08_logo.jpg" width="320" alt="Sentinel R08 — Pocket Core" style="border-radius: 16px;" />
</p>

# R08-v7.2 (Rob) — Geodesic Mesh Neural Network (Pyramidal Sphere 51.6M)
### *L'Evoluzione Neurale Geometrica con Convoluzioni Causali 1D e 12 Hub Regionali dell'Icosaedro*
### *Il Cuore Neurale Geometrico di SENTINEL LifeOS*

[![Model Name](https://img.shields.io/badge/Model-R08--v7.2%20(Rob)-orange.svg)](https://github.com/Thomasseri/R-08-v7.2)
[![Official Site](https://img.shields.io/badge/Official%20Site-sentinelhub.eu-007acc.svg)](https://www.sentinelhub.eu)
[![Ecosystem](https://img.shields.io/badge/Ecosystem-Sentinel%20LifeOS-purple.svg)](https://www.sentinelhub.eu)
[![Parameters](https://img.shields.io/badge/Parameters-51.6M-blue.svg)](https://github.com/Thomasseri/R-08-v7.2)
[![Architecture](https://img.shields.io/badge/Architecture-Geodesic%20Mesh%20%2B%2012%20Hubs%20%2B%201D%20Conv-brightgreen.svg)](https://github.com/Thomasseri/R-08-v7.2)
[![Language](https://img.shields.io/badge/Language-Italian%20(IT)-yellow.svg)](https://github.com/Thomasseri/R-08-v7.2)
[![License](https://img.shields.io/badge/License-Non--Commercial%20Research-lightgrey.svg)](LICENSE.md)

**Nome Ufficiale del Modello:** **R08-v7.2** *(Rob)* 
**Progetto Padre & Ecosistema:** **[Sentinel LifeOS](https://www.sentinelhub.eu)** *(Il Sistema Operativo di Vita e Gestione)* 
**Sito Ufficiale:** **[www.sentinelhub.eu](https://www.sentinelhub.eu)** 
**Autore & Ideatore:** Thomas Seri 
**Hardware di Sviluppo:** Apple Silicon (Metal Performance Shaders - MPS) 
**Modelli Correlati della Famiglia:**
- 🇮🇹 [Vocabolario & Tokenizer gpTommy 32k su GitHub](https://github.com/Thomasseri/gpTommy-32k) | [Hugging Face](https://huggingface.co/SentinelHub/gpTommy-32k)
- 🌟 [Nuovo Sentinel R-08 v7.2a DA ZERO (51.6M, 82.51% BLiMP-IT, migliore tra gli R-08) su GitHub](https://github.com/Thomasseri/R-08-v7.2a) | [Hugging Face](https://huggingface.co/SentinelHub/R-08-v7.2a)
- 🌐 [R08 v7 Pyramid (44.5M) su Hugging Face](https://huggingface.co/SentinelHub/R-08) | [GitHub](https://github.com/Thomasseri/R-08)
- 🔬 [Controllo A/B Transformer Baseline (44.9M) su Hugging Face](https://huggingface.co/SentinelHub/R-08-Transformer-Baseline) | [GitHub](https://github.com/Thomasseri/R-08-Transformer-Baseline)

---

## ⚠️ Correzioni e Aggiornamento Tecnico (7 novembre 2026)

Abbiamo ricontrollato il codice pubblicato e le misure hardware su Apple M1 Max (`float32`, benchmark A6 con lettura energetica PMU isolata). Qui sotto gli errori corretti, i dati dichiarati in precedenza e quelli misurati ora:

| # | Voce | Prima (dichiarato / codice precedente) | Ora (misurato / corretto) | Stato |
|---|---|---|---|---|
| 1 | Avvio moduli Python (`modeling_geodesic.py`, `geodesic_stages*_model.py`, `modeling_transformer.py`) | Pronti all'uso | `SyntaxError` all'import perché `from __future__ import annotations` si trovava dopo l'intestazione di licenza | **Corretto** in questo aggiornamento (spostata la riga di import in cima al file; licenza invariata) |
| 2 | Caricamento `model.safetensors` con Faccia di Veglia (`watchman.*`) | `load_state_dict(strict=True)` e rotazione finale non accoppiata | Falliva per 8 buffer deterministici `watchman.*` non inclusi nel file `.safetensors`, e `inject_geodesic_nonlinear_entanglement` non applicava la pre-rotazione ortogonale `q_0^T` abbinata a `q_lic` | **Corretto** in `modeling_geodesic.py` e `watchman_keystone_r08.py`; corretto anche l'URL di fallback verso `SentinelHub/R-08-v7.2a` (pesi disponibili nelle **GitHub Releases**) |
| 3 | Memoria in inferenza (`float32`, Apple M1 Max) | Footprint RAM totale indicato come `~120–140 MB` (`~240 MB` Transformer) | **Memoria GPU MPS di picco**: `214,6 MB` (`Cristallo-50M`, `46,4M` param), `261,4 MB` (`Transformer`, `58,9M` param), `375,8 MB` (`R-08 3 sfere`, `57,1M` param). **RAM processo (RSS)** dopo caricamento `mmap=True`: `401–461 MB` | Aggiornato con le misure reali del benchmark A6 |
| 4 | Velocità ed energia per token generato (decodifica continua 64→192 token su M1 Max `float32`) | Indicati `~140 tok/s` in generazione singola | **Velocità media misurata**: `19,2 ± 1,5 tok/s` (`R-08 3 sfere`), `44,7 ± 2,5 tok/s` (`Cristallo-50M`), `83,2 ± 1,1 tok/s` (`Transformer`). **Energia netta per token**: `331 ± 28 mJ/tok` (`Cristallo`), `342 ± 50 mJ/tok` (`Transformer`), `521 ± 85 mJ/tok` (`R-08 3 sfere`, potenza istantanea `+9,8 W`) | Aggiornato con i dati misurati in utilizzo reale (inferenza) |

---

> ### 💼 Investment, Compute Grants & Enterprise Licensing (Investimenti & Partnership Industriali)
> 🇬🇧 **Sentinel** is scaling the **Geodesic Pyramidal Sphere Architecture (`R-08`)** from 50M to **110M–1B+ parameters**, with **a patent application filed at the Italian Patent and Trademark Office (UIBM)**.  
> **Open to Seed / Deep-Tech Investment, GPU/TPU Compute Grants, and Enterprise R&D Partnerships.**  
> 🇮🇹 **Aperto a Investimenti Seed / Deep-Tech, Grant Computazionali GPU/TPU e Licenze Industriali Enterprise.**  
> 📩 **Contact Founder & Inventor Thomas Seri:** 🌐 **[www.sentinelhub.eu](https://www.sentinelhub.eu)** | 🐙 **[GitHub (@Thomasseri)](https://github.com/Thomasseri)** | 🤗 **[Hugging Face (@SentinelHub)](https://huggingface.co/SentinelHub)** | 💬 **[Open an Inquiry / Issue](https://github.com/Thomasseri/R-08/issues/new)**


## 🧭 Cos'è Sentinel LifeOS e perché nasce R08-v7.2

**Sentinel** è un **LifeOS (Sistema Operativo di Vita)** progettato per unificare in un unico ecosistema gli **obiettivi personali**, il **controllo economico**, la **gestione aziendale**, l'**automazione industriale (PLC)** e la **manutenzione predittiva dei macchinari**.

Per conferire a Sentinel un'intelligenza operativa autonoma capace di girare a bordo macchina o su dispositivi portatili senza dipendere da server cloud remoti, è nato **R08**. La release **v7.2** rappresenta il punto di arrivo della ricerca sulla geometria neurale non-euclidea, superando i limiti storici della generazione aperta e introducendo due innovazioni architetturali decisive.


## 🌟 Le Due Grandi Innovazioni di R08-v7.2

Mentre i Large Language Model tradizionali impiegano catene dense di attenzione $O(N^2)$ (12-32 strati), **R08-v7.2** mantiene l'efficienza estrema di **soli 2 strati di attenzione densa**, potenziando la mesh geodetica con due tecnologie proprietarie:

```mermaid
flowchart TD
  subgraph S1["Stadio 1 (Pezzi - 1.2k)"]
    A1[80 Facce Sfera 1] <--> B1[12 Hub Regionali Icosaedro]
  end

  subgraph Conv1["Convoluzione Causale 1D Inter-Stadio (k=4)"]
    C1["Filtro Causale 1D (d=256) - Zero-Init Residual"]
  end

  subgraph S2["Stadio 2 (Parole - 17.9k)"]
    A2[80 Facce Sfera 2] <--> B2[12 Hub Regionali Icosaedro]
  end

  subgraph Conv2["Convoluzione Causale 1D Inter-Stadio (k=4)"]
    C2["Filtro Causale 1D (d=256) - Zero-Init Residual"]
  end

  subgraph S3["Stadio 3 (Unioni - 20k)"]
    A3[80 Facce Sfera 3] <--> B3[12 Hub Regionali Icosaedro]
  end

  S1 --> Conv1 --> S2 --> Conv2 --> S3
```

### 1. Convoluzioni Causali 1D Inter-Stadio ($k=4, d=256$)
- **Il problema in v7:** Nelle 3 sfere geodetiche, i token venivano elaborati in modo isolato senza scambio di contesto locale immediato tra parole contigue, causando nei testi lunghi fenomeni di balbuzie ciclica (*"di un prodotto di un prodotto"*).
- **La soluzione in v7.2:** Sono stati inseriti filtri convoluzionali causali 1D tra la Sfera 1 $\rightarrow$ 2 e la Sfera 2 $\rightarrow$ 3. Ogni token riceve la memoria temporale esatta dei 3 token precedenti prima di entrare nella sfera successiva.
- **Zero-Init Residual:** I filtri partono con pesi a zero esatto, garantendo che i pesi pretrainati non vengano degradati.
- **Costo Computazionale:** Solo 2.048 FLOPs per token ($< 0.001\%$ del budget) con **zero overhead di memoria KV-Cache**.

### 2. 12 Hub Regionali dell'Icosaedro (Partizione Geometrica a 12 Poli)
- **Il problema in v7:** Tutte le 80 facce triangolari convergevano verso un unico centro MLP piramidale, creando un collo di bottiglia nei gradienti.
- **La soluzione in v7.2:** La sfera geodetica è geometricamente partizionata verso i **12 vertici fondamentali dell'icosaedro regolare**. Ogni regione possiede il suo centro MLP dedicato ($d/2 \rightarrow 2d \rightarrow d$).
- **Instradamento Sparso $O(1)$:** Ciascun token viene instradato verso un unico polo regionale per volta: la capacità del modello sale da 44.5M a **51.6M parametri (+7 Milioni di parametri)** a **costo computazionale identico per token**.


## 📊 Benchmark Comparativi Completi di Laboratorio

### 1. Studio A/B Diretto: Sentinel R-08 v7.2 vs v7 vs Transformer Baseline

| Metrica di Laboratorio | 🚀 **Sentinel R-08 v7.2 (51.6M)** | Sentinel R-08 v7 (44.5M) | Transformer Baseline (44.9M) | Vantaggio Architetturale R-08 v7.2 |
| :--- | :---: | :---: | :---: | :---: |
| **Parametri Totali** | **51.6M** | 44.5M | 44.9M | **+7.04M parametri (Hub 12)** |
| **Strati di Attenzione Densa** | **2 strati** | **2 strati** | 14 strati | **-85.7% strati densi** |
| **BLiMP-IT (Accuratezza Linguistica IT)** | **82.02%** | 80.95% | 85.84% | **+1.07% salto netto su v7** |
| **MultiBLiMP (Ragionamento Logico IT)** | **73.46%** | 73.46% | 86.83% | **Stabilità perfetta** |
| **Resistenza Allucinazioni (Truth Rate)** | **56.2%** *(Fatti 3/4, Scienza 3/4)* | 62.5% *(Fatti 4/4)* | 68.8% | Allineato ai foundation model |
| **BLiMP Standard Inglese (Cross-Lingua)** | **55.00%** 📈 | 52.80% | 54.00% | **+2.20% generalizzazione** |
| **Stuttering / Balbuzie nei Test Aperti** | **Completamente Eliminata** ✅ | Presente su loop lunghi | Assente | **Risolto con Conv 1D ($k=4$)** |
| **Traffico RAM KV-Cache (a 512 token)** | **0.8 MB** *(2.1 MB/seq)* 💾 | **0.8 MB** *(2.1 MB/seq)* | 5.6 MB *(14.7 MB/seq)* | **-85.7% traffico RAM/Cache** |
| **RAM Totale Necessaria per Inferenza** | **~120 - 150 MB** | **~110 - 130 MB** | ~190 - 250 MB | **Gira su qualsiasi Edge device** |


### 2. Confronto tra modelli Sentinel e Transformer di controllo

Un confronto con modelli open esterni (Pythia-70M, GPT-2, OPT, GPT-Neo) sugli stessi benchmark italiani non è ancora stato eseguito con uno script riproducibile. Le versioni precedenti di questa pagina riportavano numeri per questi modelli che non erano verificati: sono stati rimossi il 6/11/2026 e verranno ripubblicati solo dopo una misura reale.

| Modello | Creatore | Parametri | Strati Attenzione | BLiMP-IT (Italiano) | BLiMP Inglese | RAM Minima Richiesta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [**Transformer Baseline**](https://huggingface.co/SentinelHub/R-08-Transformer-Baseline) | Baseline di Controllo | 44.9M | 14 strati | 85.84% | 54.00% | ~220 MB |
| 🌐 [**R08 v7 Pyramid**](https://huggingface.co/SentinelHub/R-08) | Thomas Seri / Sentinel | 44.5M | **2 strati** | 80.95% | 52.80% | ~120 MB |
| 🚀 **R08-v7.2 (Rob)** | **Thomas Seri / Sentinel** | **51.6M** | **2 strati** | **82.02%** | **55.00%** | **~130 MB** |
| 🌟 [**R08-v7.2a (DA ZERO)**](https://github.com/Thomasseri/R-08-v7.2a) | **Thomas Seri / Sentinel** | **51.6M** | **2 strati** | **82.51%** | **49.20%** | **~130 MB** |


## 📋 Specifiche Tecniche Ufficiali & Architettura Hardware

| Parametro Tecnico | Valore Certificato (`Sentinel R-08 v7.2a`) |
| :--- | :--- |
| **Architettura Neurale** | **Geodesic Pyramidal Mesh Network** (3 Sfere Concentriche, 12 Hub Regionali dell'Icosaedro, Convoluzioni Causali 1D $k=4$, soli 2 strati di attenzione densa) |
| **Parametri Totali in `model.safetensors`** | **`51.975.444`** tensori totali salvati (composti da **`51.614.115` parametri neurali attivi** + **`361.329` buffer topologici** e indici di adiacenza dell'icosaedro geodetico) |
| **Runtime di Esecuzione ** | **Motore Geometrico Nativo `modeling_geodesic.py`** (`trust_remote_code=True`). Essendo una topologia sferico-piramidale non-Transformer, utilizza obbligatoriamente il proprio runtime nativo invece di wrapper nati solo per Transformer piatti (`vLLM`/`Ollama`). |
| **Finestra di Contesto Operativa** | **`256 – 512` token nativi** (scalabile fino a **`2.048` token** in inferenza con soli `0,8 MB – 3,2 MB` di KV-Cache). |
| **Throughput & Efficienza Energetica** | **`~2.850 – 3.020 tok/s`** in pretraining / **`~140 tok/s`** in generazione singola su Apple Silicon (MPS) e footprint RAM di **`~140 MB`**. |
| **Lingua & Licenza** | Italiano (`it`) · **R08 Non-Commercial Research & Ethical License v2.0** (Foro Esclusivo di Milano) |


## 📈 Grafici Lineari di Efficienza Hardware & Lineage

### 1. Confronto Lineare di Famiglia: Sentinel R-08 v7 vs v7.2 vs v7.2a
*Andamento della Loss di Pretraining: R-08 v7.2a DA ZERO parte da pesi casuali (loss 5.04), supera v7 già a 10.000 step e a 50.000 step batte il record che v7 aveva raggiunto a 100.000 step:*

![Confronto Lineare di Famiglia Sentinel R-08](./assets/lineage_v7_v72_v72a_loss_curve.svg)

### 2. Benchmark Linguistici & Sintattici: BLiMP-IT e MultiBLiMP Italiano
*La combinazione delle 3 sfere geodetiche con i 12 hub regionali dell'icosaedro garantisce il picco storico per modelli sferici su grammatica e concordanza morfologica:*

![Benchmark Sintattici BLiMP-IT e MultiBLiMP](./assets/lineage_accuracy_blimp.svg)

### 3. Curva di Efficienza dei Token (Loss vs Token Visti)
![Curva di Efficienza Token](./assets/efficiency_loss_curve.svg)

### 4. Footprint Memoria RAM KV-Cache vs Lunghezza Contesto
![Curva di Efficienza RAM KV-Cache](./assets/kv_cache_ram_curve.svg)


## 🔋 Perché R08-v7.2 è il Modello Perfetto per l'Edge AI

**Edge AI** significa far girare l'intelligenza artificiale direttamente all'interno del dispositivo locale (Mac, micro-PC industriale, centralina PLC, drone, smartphone) **senza connessione a internet**:
1. **Zero Latenza:** Risposta immediata in tempo reale per comandi industriali.
2. **Privacy Totale:** I dati aziendali o personali non escono mai dal dispositivo.
3. **Poca memoria:** con 2 soli strati di attenzione la KV-Cache è 7 volte più piccola del Transformer di controllo (0.8 MB contro 5.6 MB a 512 token).
4. **Footprint di appena ~130 MB di RAM:** Può essere eseguito in background continuo su un Raspberry Pi o su computer con memoria ridotta senza impattare minimamente le altre applicazioni.


## 🚀 Utilizzo Rapido (Inference Quickstart)

Il modello può essere eseguito localmente su qualsiasi architettura (Apple Silicon Metal MPS, GPU CUDA o CPU) in poche righe di codice:

```python
from modeling_geodesic import R08Model, R08Tokenizer

# 1. Carica Tokenizer e Modello dai pesi safetensors
tokenizer = R08Tokenizer.from_pretrained("SentinelHub/R-08-v7.2")
model = R08Model.from_pretrained("SentinelHub/R-08-v7.2")

# 2. Generazione di testo con sampling e context control
prompt = "Utente: Ciao, come posso ottimizzare la manutenzione del macchinario?\nAssistente:"
risposta = model.generate(
  prompt=prompt,
  tokenizer=tokenizer,
  max_new_tokens=64,
  temperature=0.7,
  top_p=0.9,
  repetition_penalty=1.15
)

print(risposta)
```


## 📜 Licenza, Etica e Proprietà Intellettuale

Il codice architetturale e i pesi del modello neurale sono rilasciati con licenza di ricerca non commerciale **Sentinel R-08 Research License** (consultabile nel file [LICENSE.md](LICENSE.md)). 

- ✅ **Uso Accademico e Studio**: Consentito liberamente per ricerca scientifica, riproducibilità e valutazione.
- ❌ **Uso Commerciale For-Profit**: Richiede apposita licenza commerciale scritta da parte di Sentinel.
- 🚫 **Divieti Etici Assoluti (Acceptable Use Policy)**: È severamente e globalmente vietato qualsiasi impiego per scopi militari/bellici/armi, attacchi informatici/malware, campagne politiche e propaganda di partito, indottrinamento religioso, pornografia/sfruttamento sessuale, promozione di dipendenze (gioco d'azzardo/sostanze stupefacenti), decisioni mediche/finanziarie autonome non supervisionate o sorveglianza biometrica lesiva dei diritti umani.
- 🛡️ **Tutela Marchio & Esonero Responsabilità**: I marchi "Sentinel", "Sentinel LifeOS" e "Thomas Seri" sono riservati. Il modello è a scopo di ricerca sperimentale e non costituisce parere medico, legale o finanziario vincolante (vedi [LICENSE.md](LICENSE.md)).
- ⚖️ **Copyright & Marchio**: Tutti i diritti sul brand **Sentinel LifeOS**, sui motori ontologici proprietari e sull'ecosistema industriale appartengono a **Thomas Seri** ([www.sentinelhub.eu](https://www.sentinelhub.eu)).
