# Data Sources & Provenance (GajRaksha)

This document details the data sources, scientific citations, licensing terms, and acquisition procedures used in the **GajRaksha** platform.

---

## 1. Movebank Elephant GPS Collar Telemetry

- **Study Name**: *Elephants Kruger National Park*
- **Movebank Study ID**: `736029750`
- **Citation**:
  > Wall, J., Wittemyer, G., Klinkenberg, B., & Douglas-Hamilton, I. (2014). Movement data from "Characterizing animal movement in Kruger National Park". *Movebank Data Repository*. [https://doi.org/10.5441/001/1.0240da10](https://doi.org/10.5441/001/1.0240da10)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Parameters**: 4 collared African savannah elephants (`AM105`, `AM107`, `AM108`, `AM110`), 34,483 hourly GPS collar fixes recorded between August 2007 and October 2009.
- **Repository Inclusions**:
  > **Important**: The raw Movebank GPS telemetry downloads and individual collar location fixes are **NOT included** in this git repository to protect animal tracking data and maintain repository safety standards. Only lightweight derived model weights and evaluation summaries are version-controlled in `backend/deploy_assets/`.

### Instructions to Download Raw Movebank Data
To reproduce the training dataset locally:
1. Create a free account on [Movebank](https://www.movebank.org).
2. Accept the CC BY-NC 4.0 data terms for Study `736029750` on the Movebank web interface.
3. Configure your Movebank credentials in `.env`:
   ```bash
   MOVEBANK_USER="your_movebank_username"
   MOVEBANK_PASS="your_movebank_password"
   STUDY_ID="736029750"
   ```
4. Run the data fetcher script:
   ```bash
   python backend/movebank_fetch.py
   ```
   *Alternatively, download the published public open-access data package directly from the [Movebank Data Repository](https://datarepository.movebank.org/handle/10240/1.0240da10) and place the CSV in `data/raw/movebank_events.csv`.*

---

## 2. OpenStreetMap (OSM)

- **Source**: OpenStreetMap Contributors via Overpass API and Nominatim Geocoding API.
- **Attribution & License**: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licensed under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/).
- **Data Extracted**:
  - Administrative boundary relations for the 4 South Bengal districts: Jhargram (Relation `7923758`), Paschim Medinipur (Relation `7923759`), Bankura (Relation `7923755`), and Purulia (Relation `7923756`).
  - Rural settlement polygons and node clusters (`place=village`, `place=hamlet`, `place=suburb`).
  - Forest boundaries (`landuse=forest`, `natural=wood`).
  - Hydrological features (`natural=water`, `waterway=river`, `waterway=stream`, `waterway=canal`).
  - Road transport corridors (`highway=primary|secondary|tertiary|trunk`).

---

## 3. ESA WorldCover 10m Land Cover

- **Source**: European Space Agency (ESA) WorldCover 10m 2020 / 2021 Global Land Cover Product.
- **Citation**:
  > Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.-E., Ramoino, F., Arino, O. (2021). *ESA WorldCover 10 m 2020 v100*. [https://doi.org/10.5281/zenodo.5571923](https://doi.org/10.5281/zenodo.5571923)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Usage**: Used to compute localized cropland density (`crop_density`, Land Cover Class 40) within grid cells and village buffer perimeters.

---

## 4. NASA POWER / CHIRPS Meteorological Data

- **Source**: NASA Prediction of Worldwide Energy Resources (POWER) Agroclimatology API / Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS).
- **Access URL**: [https://power.larc.nasa.gov](https://power.larc.nasa.gov)
- **License**: Public domain / US Government Open Data policy.
- **Usage**: Daily total precipitation (mm) aggregated across 7-day and 30-day rolling windows (`rainfall_last_7_days`, `rainfall_last_30_days`) to capture moisture-driven seasonal elephant movements.

---

## 5. Wildlife Institute of India (WII) Conflict Report

- **Title**: *Human-Elephant Conflict in West Bengal: Status, Distribution and Mitigation Strategies (2000–2025)*
- **Issuing Organization**: Wildlife Institute of India (WII), Dehradun, in collaboration with the Directorate of Forests, Government of West Bengal, Ministry of Environment, Forest and Climate Change (MoEFCC).
- **Usage & Boundary**: Used strictly as an external, independent ground-truth sanity check. Historical incident hotspot names in Jhargram, Paschim Medinipur, Bankura, and Purulia were geocoded to assess spatial overlap with model-predicted high-risk grid clusters.

