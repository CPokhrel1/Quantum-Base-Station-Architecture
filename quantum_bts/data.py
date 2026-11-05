import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .config import KATHMANDU_ALT, N_MOBILE, RNG


def load_data():
    df_raw = pd.read_csv('429.csv', header=None,
        names=['radio','mcc','net','area','cell','unit','lon','lat',
               'range','samples','changeable','created','updated','averageSignal'],
        skiprows=1)
    for col in ['lon','lat','range','samples']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw = df_raw.dropna(subset=['lat','lon'])
    ktm = df_raw[
        (df_raw['lat'] >= 27.60) & (df_raw['lat'] <= 27.80) &
        (df_raw['lon'] >= 85.25) & (df_raw['lon'] <= 85.45)
    ].copy().reset_index(drop=True)

    coords = ktm[['lat','lon']].values
    db = DBSCAN(eps=0.004, min_samples=1, metric='euclidean').fit(coords)
    ktm['cluster'] = db.labels_
    clusters = ktm.groupby('cluster').agg(
        lat=('lat','mean'), lon=('lon','mean'),
        n_cells=('cell','count'),
        total_samples=('samples','sum'),
        avg_range=('range','mean'),
        gsm=('radio', lambda x: (x=='GSM').sum()),
        umts=('radio', lambda x: (x=='UMTS').sum()),
        lte=('radio', lambda x: (x=='LTE').sum()),
    ).reset_index()

    landmark_map = [
        (27.7238,85.3380,'Thamel-New Road Hub'),(27.6752,85.3764,'Lalitpur South'),
        (27.7537,85.3330,'Tokha North'),(27.7272,85.3637,'Lazimpat-Chuchepati'),
        (27.7459,85.3325,'Nagarjun Foothills'),(27.6884,85.4338,'Bhaktapur East'),
        (27.6734,85.4335,'Suryabinayak'),(27.7188,85.3070,'Kirtipur Road'),
        (27.6969,85.3530,'Koteswor'),(27.7596,85.3330,'Shivapuri Ridge'),
        (27.6925,85.3262,'Satdobato-Ekantakuna'),(27.6622,85.4492,'Banepa Corridor'),
        (27.6792,85.3495,'Gwarko'),(27.6952,85.2806,'Kirtipur'),
        (27.6718,85.3418,'Sunakothi'),(27.6760,85.4018,'Lubhu'),
        (27.6888,85.2708,'Thankot West'),(27.6875,85.2665,'Thankot Ridge'),
        (27.6878,85.2539,'Chandragiri'),(27.6918,85.2613,'Dakshinkali Road'),
        (27.6686,85.3515,'Khokana'),(27.7102,85.2998,'Kalanki'),
        (27.7223,85.2925,'Swayambhu'),(27.7142,85.2838,'Sitapaila'),
        (27.6767,85.4113,'Lubhu East'),(27.6784,85.4215,'Madhyapur Thimi'),
    ]
    def nearest_name(lat, lon):
        return min(landmark_map, key=lambda t: abs(lat-t[0])+abs(lon-t[1]))[2]
    clusters['name'] = clusters.apply(lambda r: nearest_name(r['lat'], r['lon']), axis=1)

    q75 = clusters['n_cells'].quantile(0.75)
    q25 = clusters['n_cells'].quantile(0.25)
    def get_type(n):
        if n >= q75: return 'hub'
        if n >= q25: return 'relay'
        return 'node'
    clusters['type']       = clusters['n_cells'].apply(get_type)
    clusters['altitude_m'] = KATHMANDU_ALT
    return ktm, clusters


def add_mobile_nodes(clusters, n_mobile=N_MOBILE):
    lat_min, lat_max = 27.62, 27.77
    lon_min, lon_max = 85.27, 85.43

    lat_c = clusters['lat'].mean()
    lon_c = clusters['lon'].mean()

    mob_lats = np.clip(RNG.normal(lat_c, 0.035, n_mobile), lat_min, lat_max)
    mob_lons = np.clip(RNG.normal(lon_c, 0.04,  n_mobile), lon_min, lon_max)

    headings  = RNG.uniform(0, 360, n_mobile)
    mob_types = RNG.choice(['pedestrian','vehicle','IoT'], n_mobile,
                            p=[0.5, 0.35, 0.15])

    SPEED_RANGE = {
        'pedestrian': (0.0,  6.0),
        'vehicle'   : (0.0, 80.0),
        'IoT'       : (0.0,  2.0),
    }
    speeds = np.array([
        RNG.uniform(*SPEED_RANGE[t]) for t in mob_types
    ])

    mobile_rows = []
    for i in range(n_mobile):
        mobile_rows.append({
            'mobile_id'  : f"MOB-{i+1:03d}",
            'lat'        : mob_lats[i],
            'lon'        : mob_lons[i],
            'speed_kmh'  : speeds[i],
            'heading_deg': headings[i],
            'mob_type'   : mob_types[i],
            'altitude_m' : KATHMANDU_ALT,
        })
    return pd.DataFrame(mobile_rows)
