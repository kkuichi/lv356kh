import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM, Bidirectional


def get_data(folder):
    """
    Načíta všetky NetCDF súbory zo zadaného priečinka do jedného DataFrame-u.

    Argumenty:
        folder : cesta k priečinku s .nc súbormi

    Návratova hodnota:
        df : DataFrame
    """
    files = list(Path(folder).glob('*.nc'))
    if not files:
        raise FileNotFoundError(f"V priečinku {folder} sa nenachádzajú súbory.")

    df_list = []
    for x in files:
        ds = xr.open_dataset(x)
        df_list.append(ds.to_dataframe().reset_index())

    df = pd.concat(df_list, ignore_index=True)
    return df


def preprocess_data(df, target_lat, target_lon):
    """
    Vyberie premenné potrebné na fungovanie celého kódu a odstráni chýbajúce hodnoty

    Argumenty:
        df : DataFrame z load_data
        target_lat : cieľová zemepisná šírka
        target_lon : cieľová zemepisná dĺžka
        target_time : čas v milisekundách od polnoci

    Návratova hodnota:
        df_filtered : vyčistený DataFrame
    """

    # Vyberú sa len potrebné stĺpce a odstránime riadky s chýbajúcimi hodnotami
    df_filtered = df[['tplatitude', 'tplongitude', 'tpaltitude', 'ktemp', 'H2O', 'time']].dropna().reset_index(drop=True)

    if df_filtered.empty:
        raise ValueError(f"Žiadne dáta pre šírku={target_lat} a dĺžku={target_lon}")

    print(f"Počet záznamov: {len(df_filtered):,} "
          f"z celkových {len(df):,}")

    return df_filtered


def prepare_inputs(df_filtered, target_value):
    """
    Vyberie vstupné a výstupné premenné pre model, vytvoria sa scalery a rozdelí sa na trénovaciu
    a testovaciu množinu.

    Argumenty:
        df_filtered : predspracovaný DataFrame
        target_col : názov predikovanej premennej

    Návratova hodnota:
        X_train_seq, X_test_seq : trénovacie a testovacie vstupy tvaru (N, 4, 1)
        y_train, y_test : škálované výstupy
        scaler_x, scaler_y : MinMaxScaler-y
        X_test : testovacie vstupy určené na spätnú transformáciu
        idx_test : indexy testovacích dát v pôvodnom datasete
    """

    # Vstupy modelu = zemepisná šírka, dĺžka, výška, čas
    X = df_filtered[['tplatitude', 'tplongitude', 'tpaltitude', 'time']].values

    # Výstup modelu = hodnota predikovanej premennej
    y = df_filtered[target_value].values.reshape(-1, 1)

    # Škálujeme vstupy aj výstup do rozsahu [0, 1]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Rozdelíme indexy namiesto priameho rozdelenia dát,
    # aby sme si uchovali informáciu o pôvodnej pozícii každého riadku
    indices = np.arange(len(X_scaled))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = X_scaled[idx_train]
    X_test  = X_scaled[idx_test]
    y_train = y_scaled[idx_train]
    y_test  = y_scaled[idx_test]

    # Reshape na tvar (N, 4, 1)
    X_train_seq = X_train.reshape(-1, 4, 1)
    X_test_seq  = X_test.reshape(-1, 4, 1)

    return X_train_seq, X_test_seq, y_train, y_test, scaler_x, scaler_y, X_test, idx_test


def closest_event(X_test_real, y_test, idx_test, target_lat, target_lon):
    """
    V testovacej množine sa nájde event, ktorého priemerná geografická
    poloha je najbližšia k zadanej polohe kde Každý event obsahuje 394 výškových hladín.
    Argumenty:
        X_test_real : reálne hodnoty vstupov testovacej množiny
        y_test : reálne hodnoty výstupu testovacej množiny
        idx_test : pôvodné indexy testovacích dát
        target_lat : cieľová zemepisná šírka
        target_lon : cieľová zemepisná dĺžka

    Návratova hodnota:
        closest_alts   : výškové hladiny najbližšieho eventu
        closest_values : hodnoty predikovanej premennej najbližšieho eventu
        lat_event      : priemerná zemepisná šírka najbližšieho eventu
        lon_event      : priemerná zemepisná dĺžka najbližšieho eventu
    """
    
    # Odvodíme eventu z pôvodného indexu
    event_ids_test = idx_test // 394

    # Vzdialenosť každého riadku od zadanej polohy
    dist = np.sqrt(
        (X_test_real[:, 0] - target_lat) ** 2 +
        (X_test_real[:, 1] - target_lon) ** 2
    )

    # Zpriemerovanie vzdialenosi pre každý event a nájdenie najbližšieho
    df_dist = pd.DataFrame({'dist': dist, 'event_id': event_ids_test})
    closest_event_id = df_dist.groupby('event_id')['dist'].mean().idxmin()

    mask = event_ids_test == closest_event_id
    closest_alts   = X_test_real[mask, 2] 
    closest_values = y_test[mask].flatten()

    # Zoradíme podľa výšky
    sort_idx = np.argsort(closest_alts)
    closest_alts   = closest_alts[sort_idx]
    closest_values = closest_values[sort_idx]

    lat_event = X_test_real[mask, 0].mean()
    lon_event = X_test_real[mask, 1].mean()

    return closest_alts, closest_values, lat_event, lon_event


def prediction_model():
    """
    Zostaví predikčný model kombinujúci konvolučné vrstvy pre nájdenie
    lokálnych vzorov a obojsmernú LSTM vrstvu pre zachytenie
    závislostí sekvencií výškových hladín.

    Architektúra:
        2x Conv1D (64 filtrov) → Bidirectional LSTM (64) → Dropout → 3x Dense → výstup

    Návratova hodnota:
        model : skompilovaný Keras model
    """
    model = Sequential()

    # Konvolučné vrstvy
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same', input_shape=(4, 1)))
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same'))

    # Obojsmerná LSTM, ktorá spracuje sekvenciu v oboch smeroch
    model.add(Bidirectional(LSTM(64, return_sequences=False)))

    # Náhodne vypína 20 % neurónov počas trénovania
    model.add(Dropout(0.2))

    # Plne prepojené vrstvy pre finálnu predikciu
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Výstupná vrstva predikujúca jednu hodnotu
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def generate_plot(history_ktemp, history_h2o,
                 pred_prof_k, test_alts,
                 closest_alts_k, closest_vals_k,
                 pred_prof_h,
                 closest_alts_h, closest_vals_h,
                 target_lat, target_lon):
    """
    Vykreslí 4 grafy:
        [0,0] Vertikálny profil teploty pre najbližší event vs. predikcia modelu
        [0,1] Vertikálny profil koncentrácie vodnej pary pre najbližší event vs. predikcia modelu
        [1,0] Tréningová krivka teploty pre MSE stratu počas trénovania
        [1,1] Tréningová krivka koncentrácie vodnej pary pre MSE stratu počas trénovania
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Vertikálny profil teploty
    axes[0, 0].plot(closest_vals_k, closest_alts_k, color='steelblue', linewidth=2,
                    label=f'Najbližší event')
    axes[0, 0].plot(pred_prof_k, test_alts, color='red', linewidth=2.5,
                    label=f'Predikcia modelu')
    axes[0, 0].set_title(f'Vertikálny profil teploty pre (Lat={target_lat} Lon={target_lon})')
    axes[0, 0].set_xlabel('Teplota (K)')
    axes[0, 0].set_ylabel('Nadmorská výška (km)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Vertikálny profil koncentrácie vodnej pary
    axes[0, 1].plot(closest_vals_h, closest_alts_h, color='steelblue', linewidth=2,
                    label=f'Najbližší event')
    axes[0, 1].plot(pred_prof_h, test_alts, color='red', linewidth=2.5,
                    label=f'Predikcia modelu')
    axes[0, 1].set_title(f'Vertikálny profil konentrácie vodnej pary pre (Lat={target_lat} Lon={target_lon})')
    axes[0, 1].set_xlabel('H2O (ppmv)')
    axes[0, 1].set_ylabel('Nadmorská výška (km)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Tréningová krivka teploty
    axes[1, 0].plot(history_ktemp['loss'], label='Train strata')
    axes[1, 0].plot(history_ktemp['val_loss'], label='Val strata')
    axes[1, 0].set_title('Tréning teploty')
    axes[1, 0].set_xlabel('Epocha')
    axes[1, 0].set_ylabel('MSE strata')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Tréningová krivka koncentrácie vodnej pary
    axes[1, 1].plot(history_h2o['loss'], label='Train strata')
    axes[1, 1].plot(history_h2o['val_loss'], label='Val strata')
    axes[1, 1].set_title('Tréning koncentrácie vodnej pary')
    axes[1, 1].set_xlabel('Epocha')
    axes[1, 1].set_ylabel('MSE strata')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vysledky_full.png', dpi=300, bbox_inches='tight')
        
    plt.show()


if __name__ == "__main__":

    DATA_FOLDER = './data/'
    ALT_LEVELS  = 72    # počet výškových hladín pre predikované profily
    EPOCHS      = 50
    BATCH_SIZE  = 512
    TRAIN_DATA = Path('training_data')

    df = get_data(DATA_FOLDER)

    # Načítanie vstupov od používateľa
    try:
        user_lat     = float(input(f"Zadaj zemepisnú šírku (Lat) [{df.tplatitude.min():.1f} až {df.tplatitude.max():.1f}]: "))
        user_lon     = float(input(f"Zadaj zemepisnú dĺžku (Lon) [{df.tplongitude.min():.1f} až {df.tplongitude.max():.1f}]: "))
        user_hours   = int(input("Zadaj hodiny   (0–23): "))
        user_minutes = int(input("Zadaj minúty   (0–59): "))
        user_seconds = int(input("Zadaj sekundy  (0–59): "))

        # Čas prevedený na milisekundy od polnoci
        user_time = (user_hours * 3600 + user_minutes * 60 + user_seconds) * 1000

    except ValueError:
        user_lat, user_lon, user_time = 50.0, 40.0, 0

    # Získanie vyfiltrovaného datasetu
    df_filtered = preprocess_data(df, user_lat, user_lon)
    ALT_MIN = df_filtered['tpaltitude'].min()
    ALT_MAX = df_filtered['tpaltitude'].max()

    # Výšková os pre predikovaný profil
    test_alts = np.linspace(ALT_MIN, ALT_MAX, ALT_LEVELS)

    # Vstupný profil pre predikciu
    input_profile = np.column_stack([
        np.full(ALT_LEVELS, user_lat),
        np.full(ALT_LEVELS, user_lon),
        test_alts,
        np.full(ALT_LEVELS, user_time)
    ])

    print("\nTréning teploty")

    X_train_seq, X_test_seq, y_train, y_test, scaler_x_ktemp, scaler_y_ktemp, X_test_scaled, idx_test = prepare_inputs(df_filtered, 'ktemp')

    # Ak existuje uložený model, načíta sa
    if ((TRAIN_DATA / 'model_ktemp.keras').exists() and
        (TRAIN_DATA / 'history_ktemp.pkl').exists() and
        (TRAIN_DATA / 'scaler_x_ktemp.pkl').exists() and
        (TRAIN_DATA / 'scaler_y_ktemp.pkl').exists()):

        model_ktemp = load_model(TRAIN_DATA / 'model_ktemp.keras')
        with open(TRAIN_DATA / 'history_ktemp.pkl', 'rb') as f:
            data_ktemp = pickle.load(f)
        with open(TRAIN_DATA / 'scaler_x_ktemp.pkl', 'rb') as f:
            scaler_x_ktemp = pickle.load(f)
        with open(TRAIN_DATA / 'scaler_ktemp_y.pkl', 'rb') as f:
            scaler_y_ktemp = pickle.load(f)

    else:
        model_ktemp = prediction_model()
        model_ktemp.summary()

        history_ktemp = model_ktemp.fit(
            X_train_seq, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1
        )

        data_ktemp = history_ktemp.history

        model_ktemp.save(TRAIN_DATA / 'model_ktemp.keras')
        with open(TRAIN_DATA / 'history_ktemp.pkl', 'wb') as f:
            pickle.dump(data_ktemp, f)
        with open(TRAIN_DATA / 'scaler_x_ktemp.pkl', 'wb') as f:
            pickle.dump(scaler_x_ktemp, f)
        with open(TRAIN_DATA / 'scaler_y_ktemp.pkl', 'wb') as f:
            pickle.dump(scaler_y_ktemp, f)
        

    # Inverzná transformácia predikcií späť na K
    y_test_ktemp = scaler_y_ktemp.inverse_transform(y_test)
    y_pred_ktemp = scaler_y_ktemp.inverse_transform(model_ktemp.predict(X_test_seq))

    print(f"MAE ktemp: {mean_absolute_error(y_test_ktemp, y_pred_ktemp):.2f} K "
          f"MAPE = {mean_absolute_error(y_test_ktemp, y_pred_ktemp) / y_test_ktemp.mean() * 100:.2f} %")

    X_test_real_ktemp = scaler_x_ktemp.inverse_transform(X_test_scaled)

    # Výber najbližšieho eventu pre teplotu
    closest_ktemp, closest_val_ktemp, closest_lat_ktemp, closest_lon_ktemp = closest_event(
        X_test_real_ktemp, y_test_ktemp, idx_test, user_lat, user_lon
    )

    # Predikcia vertikálneho profilu teploty
    input_prof_scaled_k = scaler_x_ktemp.transform(input_profile).reshape(-1, 4, 1)
    pred_ktemp = scaler_y_ktemp.inverse_transform(model_ktemp.predict(input_prof_scaled_k)).flatten()

    time.sleep(600)

    print("\nTréning koncentrácie vodnej pary")

    X_train_seq, X_test_seq, y_train, y_test, scaler_x_h2o, scaler_y_h2o, X_test_scaled, idx_test = prepare_inputs(df_filtered, 'H2O')

    if ((TRAIN_DATA / 'model_h2o.keras').exists() and
        (TRAIN_DATA / 'history_h2o.pkl').exists() and
        (TRAIN_DATA / 'scaler_x_h2o.pkl').exists() and
        (TRAIN_DATA / 'scaler_y_h2o.pkl').exists()):

        model_h2o = load_model(TRAIN_DATA / 'model_h2o.keras')
        with open(TRAIN_DATA / 'history_h2o.pkl', 'rb') as f:
            data_h2o = pickle.load(f)
        with open(TRAIN_DATA / 'scaler_x_h2o.pkl', 'rb') as f:
            scaler_x_h2o = pickle.load(f)
        with open(TRAIN_DATA / 'scaler_y_h2o.pkl', 'rb') as f:
            scaler_y_h2o = pickle.load(f)

    else:
        model_h2o = prediction_model()
        model_h2o.summary()

        history_h = model_h2o.fit(
            X_train_seq, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1
        )

        data_h2o = history_h.history

        model_h2o.save(TRAIN_DATA / 'model_h2o.keras')
        with open(TRAIN_DATA / 'history_h2o.pkl', 'wb') as f:
            pickle.dump(data_h2o, f)
        with open(TRAIN_DATA / 'scaler_x_h2o.pkl', 'wb') as f:
            pickle.dump(scaler_x_h2o, f)
        with open(TRAIN_DATA / 'scaler_y_h2o.pkl', 'wb') as f:
            pickle.dump(scaler_y_h2o, f)
        

    # Transformácia späť na ppmv
    y_test_h2o = scaler_y_h2o.inverse_transform(y_test)
    y_pred_h2o = scaler_y_h2o.inverse_transform(model_h2o.predict(X_test_seq))

    print(f"MAPE: ({mean_absolute_error(y_test_h2o, y_pred_h2o) / y_test_h2o.mean() * 100:.2f} %)")

    X_test_real_h2o = scaler_x_h2o.inverse_transform(X_test_scaled)

    # Výber najbližšieho eventu pre koncentráciu vodnej pary
    closest_h2o, closest_vals_h2o, closest_lat_h2o, closest_lon_h2o = closest_event(
        X_test_real_h2o, y_test_h2o, idx_test, user_lat, user_lon
    )

    # Predikcia vertikálneho profilu koncentrácie vodnej pary
    input_scaled_h2o = scaler_x_h2o.transform(input_profile).reshape(-1, 4, 1)
    pred_h2o = scaler_y_h2o.inverse_transform(model_h2o.predict(input_scaled_h2o)).flatten()

    generate_plot(
        data_ktemp, data_h2o,
        pred_ktemp, test_alts,
        closest_ktemp, closest_val_ktemp, pred_h2o,
        closest_h2o, closest_vals_h2o,
        user_lat, user_lon
    )
