# Data Sources

Seastate supports multiple oceanographic data sources from Copernicus Marine Service (CMEMS) and EUMETSAT.

## Copernicus Marine Service (CMEMS)

### Sea Surface Height (SSH)

**Module**: `seastate.data_sources.cmems_ssh`

**Product**: SEALEVEL_GLO_PHY_L4_NRT_008_046

| Property | Value |
|----------|-------|
| Resolution | 0.125° (~14 km) |
| Temporal | Daily |
| Variables | SLA, ADT, UGOS, VGOS |
| Coverage | Global |

```python
from seastate.data_sources import cmems_ssh

# Open dataset for a specific date
ds = cmems_ssh.open_dataset(dtm="2026-01-15")

# Access sea level anomaly
sla = ds.sla
print(f"SLA range: {float(sla.min()):.2f} to {float(sla.max()):.2f} m")

# Open as Satpy scene
scn = cmems_ssh.open_scene(dtm="2026-01-15")
```

**Variables**:

- `sla`: Sea Level Anomaly (m)
- `adt`: Absolute Dynamic Topography (m)
- `ugos`: Geostrophic velocity U component (m/s)
- `vgos`: Geostrophic velocity V component (m/s)

### Sea Surface Temperature (OSTIA)

**Module**: `seastate.data_sources.ostia`

**Product**: SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001

| Property | Value |
|----------|-------|
| Resolution | 0.05° (~5 km) |
| Temporal | Daily |
| Variables | analysed_sst |
| Coverage | Global |

```python
from seastate.data_sources import ostia

# Open dataset
ds = ostia.open_dataset(dtm="2026-01-15")

# Access SST (convert from Kelvin to Celsius)
sst_celsius = ds.analysed_sst - 273.15
print(f"SST range: {float(sst_celsius.min()):.1f} to {float(sst_celsius.max()):.1f} °C")
```

### Chlorophyll-a (GlobColour)

**Module**: `seastate.data_sources.globcolour`

**Product**: OCEANCOLOUR_GLO_BGC_L3_NRT_009_101

| Property | Value |
|----------|-------|
| Resolution | 4 km |
| Temporal | Daily |
| Variables | CHL |
| Coverage | Global |

```python
from seastate.data_sources import globcolour

# Open dataset
ds = globcolour.open_dataset(dtm="2026-01-15")

# Access chlorophyll (mg/m³)
chl = ds.CHL
print(f"Chl range: {float(chl.min()):.3f} to {float(chl.max()):.1f} mg/m³")
```

!!! note "Log Transformation"
    Chlorophyll data is typically displayed on a logarithmic scale due to its wide dynamic range (0.01 - 100 mg/m³).

## EUMETSAT Data Store

### Sentinel-3 OLCI Level-2

**Module**: `seastate.data_sources.olci_L2`

**Product**: EO:EUM:DAT:0407

| Property | Value |
|----------|-------|
| Resolution | 300 m |
| Temporal | Per-swath |
| Variables | chl_oc4me, chl_nn |
| Coverage | Swath-based |

```python
from seastate.data_sources import olci_L2

# Retrieve swaths for a date
olci_L2.retrieve(dtm="2026-01-15")

# List downloaded swaths
swaths = olci_L2.swathlist(dtm="2026-01-15")
print(f"Found {len(swaths)} swaths")

# Extract and load scenes
scenes = olci_L2.extract_swath_scenes(dtm="2026-01-15")
```

!!! warning "Credentials Required"
    EUMETSAT data access requires credentials stored in `~/.eumdac/credentials`.

## Data Caching

All data sources implement local caching:

- Data is downloaded once and stored in `{data_dir}/{source}/`
- Subsequent requests use cached files
- Use `force=True` to re-download data

```python
# Use cached data (default)
ds = cmems_ssh.open_dataset(dtm="2026-01-15")

# Force re-download
cmems_ssh.retrieve(dtm="2026-01-15", force=True)
ds = cmems_ssh.open_dataset(dtm="2026-01-15")
```

## Configuring Data Bounds

Data retrieval is limited to the geographic bounds defined in configuration:

```toml
[default]
lat1 = 45.0   # Southern boundary
lat2 = 65.0   # Northern boundary
lon1 = -10.0  # Western boundary
lon2 = 30.0   # Eastern boundary
```

This reduces download size and processing time.

## Common Issues

### Authentication Errors

Ensure credentials are properly configured:

```toml
# .secrets.toml
[default]
cmems_login = "your_username"
cmems_password = "your_password"
```

### Missing Data

Some products may not be available for recent dates (NRT latency) or historical dates (archive limits).

```python
from copernicusmarine import CoordinatesOutOfDatasetBounds

try:
    ds = cmems_ssh.open_dataset(dtm="2020-01-01")
except CoordinatesOutOfDatasetBounds:
    print("Data not available for this date")
```

### Network Timeouts

Large data requests may timeout. Consider:

- Reducing geographic bounds
- Using parallel downloads where supported
- Retrying failed downloads
