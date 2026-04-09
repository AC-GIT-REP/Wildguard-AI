import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# Realistic park definitions (approximate centroids and bounds)
PARKS = {
    "Kaziranga National Park": {
        "country": "India",
        "lat": 26.5775, "lon": 93.1711, "radius": 0.15,
        "species": ["Indian Rhino", "Bengal Tiger", "Asian Elephant"],
        "base_risk": 0.7
    },
    "Bandipur Tiger Reserve": {
        "country": "India",
        "lat": 11.6664, "lon": 76.6288, "radius": 0.12,
        "species": ["Bengal Tiger", "Asian Elephant", "Indian Leopard"],
        "base_risk": 0.5
    },
    "Jim Corbett National Park": {
        "country": "India",
        "lat": 29.5300, "lon": 78.7747, "radius": 0.18,
        "species": ["Bengal Tiger", "Asian Elephant"],
        "base_risk": 0.6
    },
    "Kruger National Park": {
        "country": "South Africa",
        "lat": -23.9884, "lon": 31.5590, "radius": 0.40,
        "species": ["White Rhino", "African Elephant", "Pangolin"],
        "base_risk": 0.85
    },
    "Serengeti National Park": {
        "country": "Tanzania",
        "lat": -2.3333, "lon": 34.8333, "radius": 0.35,
        "species": ["African Elephant", "African Lion", "Black Rhino"],
        "base_risk": 0.75
    }
}

INCIDENT_TYPES = ["Snare Discovered", "Carcass Found", "Gunshots Reported", "Illegal Camp Suspicion", "Footprints/Tracks"]
MOON_PHASES = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]

# Poachers often strike near Full Moons (visibility) and use snares more frequently
def get_moon_phase(date):
    # Simplistic deterministic moon phase proxy for mock data
    days = (date - datetime(2020, 1, 1)).days
    cycle = (days % 29.53)
    if cycle < 3 or cycle > 26: return "New Moon"
    elif 13 < cycle < 17: return "Full Moon"
    elif cycle <= 13: return "First Quarter"
    else: return "Last Quarter"

def generate_poaching_data(num_records=1500, years_back=5):
    records = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    for _ in range(num_records):
        # Pick a park based on base risk
        park_name = random.choices(
            list(PARKS.keys()), 
            weights=[p["base_risk"] for p in PARKS.values()]
        )[0]
        park = PARKS[park_name]
        
        # Determine date (add an intentional recent spike to Kruger and Kaziranga)
        if park_name in ["Kruger National Park", "Kaziranga National Park"] and random.random() < 0.2:
            # 20% of incidents for high-risk parks happen in the last 60 days (SPIKE!)
            days_ago = random.randint(0, 60)
        else:
            days_ago = random.randint(0, 365 * years_back)
            
        incident_date = end_date - timedelta(days=days_ago)
        moon = get_moon_phase(incident_date)
        
        # Determine incident type (Full moon -> Gunshots more likely, New moon -> Snares)
        incident_weights = [0.4, 0.1, 0.2, 0.1, 0.2]
        if moon == "Full Moon":
            incident_weights = [0.2, 0.2, 0.5, 0.05, 0.05]
            
        itype = random.choices(INCIDENT_TYPES, weights=incident_weights)[0]
        
        # Geospatial scatter around the park centroid
        jitter_lat = np.random.normal(0, park["radius"] / 3)
        jitter_lon = np.random.normal(0, park["radius"] / 3)
        
        lat = park["lat"] + jitter_lat
        lon = park["lon"] + jitter_lon
        
        species = random.choice(park["species"])
        
        records.append({
            "date": incident_date.strftime("%Y-%m-%d"),
            "park_name": park_name,
            "country": park["country"],
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "species_targeted": species,
            "incident_type": itype,
            "moon_phase": moon
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="date", ascending=False).reset_index(drop=True)
    
    # Save to data dir
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "poaching_incidents.csv", index=False)
    print(f"Generated {len(df)} realistic poaching records inside 'data/poaching_incidents.csv'.")

if __name__ == "__main__":
    generate_poaching_data(num_records=2500, years_back=6)
