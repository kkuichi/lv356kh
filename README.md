# Predikcia atmosférických vertikálnych profilov vybraných veličín rádiometra SABER

Predikčný model na odhadovanie atmosférických vertikálnych profilov teploty a koncentrácie vodnej pary pomocou hybridnej hlbokej neurónovej siete Conv1D-BiLSTM trénovanej na meraniach satelitu SABER.

## Prehľad

Tento projekt rieši priestorové a časové medzery v satelitných atmosferických meraniach. Prístroj SABER na palube satelitu TIMED meria vertikálne profily iba pozdĺž trasy orbity satelitu. Tento model odhaduje vertikálne profily pre ľubovoľné polohy a časy na základe historických dát SABER v jednom mesiaci.

**Predikované veličiny:**
- Teplota [K]
- Koncentrácia vodnej pary [ppmv]

## Architektúra modelu

Hybridná neurónová sieť **Conv1D-BiLSTM**:
- 2× Conv1D vrstvy (64 filtrov, veľkosť jadra 2, ReLU)
- Obojsmerná LSTM (64)
- Dropout (0,2)
- 3× plne prepojené vrstvy Dense (128 → 64 → 32, ReLU)
- Výstupná vrstva (1)

**Vstupné premenné:** zemepisná šírka, zemepisná dĺžka, nadmorská výška, čas  
**Výstup:** jedna atmosferická hodnota pre každú výškovú hladinu v mriežke

## Dáta

- **Zdroj:** dáta SABER úrovne 2A
- **Formát:** NetCDF (.nc)
- **Použité obdobie:** mesiac Január z rokov 2021-2025
Z dôvodu veľkosti jednotlivých súborov je potrebné si stiahnuť dáta zo [SABER-a](https://saber.gats-inc.com/custom.php)
prihlásením sa do custom toolu a následného vybrania mesiaca a dní mesiaca Január. V dolnej časti sú štyri dropdown menu, kde je potrebné si vybrať premenné ktemp a H2O.
Stiahnuté `.nc` súbory umiestnite do priečinka `data/`.

## Štruktúra projektu

```
├── model.py        # Hlavný skript
├── data/           # Vstupné NetCDF súbory
├── training_data/  # Uložený model, história trénovania a scalery
└── README.md
```

## Inštalácia

```bash
pip install numpy
pip install pandas
pip install xarray
pip install scikit-learn
pip install tensorflow
pip install matplotlib
pip install pickle5
```

## Použitie

```bash
python model.py
```

Po zapnutí zadajte:
- Zemepisnú šírku
- Zemepisnú dĺžku
- Čas v hodinách, minútach, sekundách

## Postup behu skriptu

1. Načítanie dát zo všetkých `.nc` súborov v priečinku `data/`
2. Zadanie vstupných hodnôt používateľom
3. Tréning modelu pre teplotu alebo načítanie uloženého modelu
4. Predikcia vertikálneho profilu teploty
5. Pauza 10 minút (Thermal throttling)
6. Tréning modelu pre vodnú paru alebo načítanie uloženého modelu
7. Predikcia vertikálneho profilu vodnej pary
8. Vizualizácia a uloženie výsledkov

## Uložené súbory

Po trénovaní sa automaticky uložia nasledujúce súbory do priečinka `training_data/`:

| Súbor | Popis |
|-------|-------|
| `model_ktemp.keras` | Natrénovaný model teploty |
| `model_h2o.keras` | Natrénovaný model vodnej pary |
| `scaler_x_ktemp.pkl` | Scaler modelu teploty |
| `scaler_y_ktemp.pkl` | Scaler modelu teploty |
| `scaler_x_h2o.pkl` | Scaler modelu koncentrácie vodnej pary |
| `scaler_y_h2o.pkl` | Scaler modelu koncentrácie vodnej pary |
| `history_ktemp.pkl` | História trénovania teploty |
| `history_h2o.pkl` | História trénovania vodnej pary |

Ak tieto súbory už existujú, model sa načíta namiesto opätovného trénovania.

## Výstupné grafy

Po predikcii sa automaticky uložia nasledujúce obrázky:

| Súbor | Popis |
|-------|-------|
| `vysledky_full.png` | Všetky 4 grafy v jednom obrázku (2×2) |

Rozloženie grafov 2×2:
- **Vľavo hore:** Predikovaný vs. najbližší reálny vertikálny profil teploty
- **Vpravo hore:** Predikovaný vs. najbližší reálny vertikálny profil vodnej pary
- **Vľavo dole:** Tréningová krivka MSE straty pre teplotu
- **Vpravo dole:** Tréningová krivka MSE straty pre vodnú paru

## Nastavenia trénovania

| Parameter | Hodnota | Popis |
|-----------|---------|-------|
| `EPOCHS` | 50 | Počet trénovacích epôch |
| `BATCH_SIZE` | 512 | Veľkosť dávky |
| `ALT_LEVELS` | 72 | Počet výškových hladín predikovaného profilu |
| `validation_split` | 0,1 | Podiel validačnej množiny z trénovacej |
| `test_size` | 0,2 | Podiel testovacej množiny |
| `random_state` | 42 | Fixné rozdelenie dát |

## Vyhodnotenie

Modely sú vyhodnocované pomocou:
- **MAE** (stredná absolútna chyba) vo fyzikálnych jednotkách (K alebo ppmv)
- **MAPE** (stredná absolútna percentuálna chyba) v %
- Vizuálne porovnanie predikovaného vertikálneho profilu s najbližším reálnym eventom z testovacej množiny

## Poznámka

Medzi trénovaním ktemp a H2O modelu je zaradená 10-minútová pauza (`time.sleep(600)`) z dôvodu tepelného zaťaženia CPU. Ak sa oba modely spúšťaju s dostatočným časovým odstupom, pauzu môžete odstrániť.
