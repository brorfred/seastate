# Configuration

Seastate uses [Dynaconf](https://www.dynaconf.com/) for configuration management, supporting multiple configuration files and environment-based settings.

## Configuration File Locations

Settings are loaded from multiple locations in order of increasing priority:

1. `/etc/seastate/settings.toml` - Global system settings
2. `/etc/seastate/.secrets.toml` - Global secrets
3. `~/.config/seastate/settings.toml` - User settings
4. `~/.config/seastate/.secrets.toml` - User secrets
5. `./settings.toml` - Project settings
6. `./.secrets.toml` - Project secrets
7. `SEASTATE_SETTINGS_FILE_FOR_DYNACONF` - Environment variable path

Later files override earlier ones.

## Configuration Options

### Basic Settings

Create a `settings.toml` file:

```toml
[default]
# Geographic bounds for data retrieval
lat1 = 45.0
lat2 = 65.0
lon1 = -10.0
lon2 = 30.0

# Zoom levels for tile generation
zoom_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Maximum days to keep in tile archive
max_tile_days = 14

# Directory paths
data_dir = "/path/to/data"
tile_dir = "/path/to/tiles"

# Cruise identification
cruise_name = "my_cruise_2026"

# Remote sync settings
remote_sync = true
remote_server = "user@server.example.com"
remote_tile_dir = "/var/www/tiles"
remote_html_dir = "/var/www/html"
remote_url = "https://tiles.example.com"
```

### Secrets Configuration

Create a `.secrets.toml` file for credentials:

```toml
[default]
# Copernicus Marine Service credentials
cmems_login = "your_username"
cmems_password = "your_password"
```

!!! warning "Security"
    Never commit `.secrets.toml` to version control. Add it to your `.gitignore`.

## Environment Variables

You can also set configuration via environment variables with the `SEASTATE_` prefix:

```bash
export SEASTATE_LAT1=45.0
export SEASTATE_LAT2=65.0
export SEASTATE_CMEMS_LOGIN=your_username
```

## Multiple Environments

Define different environments in your configuration:

```toml
[default]
lat1 = -90
lat2 = 90

[development]
lat1 = 45
lat2 = 55
remote_sync = false

[production]
lat1 = 40
lat2 = 70
remote_sync = true
```

Switch environments in code:

```python
from seastate.config import change_env

# Switch to development environment
change_env("development")
```

## Verifying Configuration

```python
from seastate import config

# Print all loaded settings
print("Configuration files:", config.settings.loaded_files)
print("Current settings:", dict(config.settings))

# Access specific settings
print("Latitude range:", config.settings.lat1, "-", config.settings.lat2)
print("Tile directory:", config.settings.tile_dir)
```

## Example Complete Configuration

```toml
# settings.toml
[default]
# Geographic bounds (Baltic Sea example)
lat1 = 53.0
lat2 = 66.0
lon1 = 9.0
lon2 = 31.0

# Tile settings
zoom_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
max_tile_days = 30

# Local directories
data_dir = "/data/cruise/raw"
tile_dir = "/data/cruise/tiles"

# Cruise info
cruise_name = "baltic_2026"

# Remote sync
remote_sync = true
remote_server = "tiles.oceanserver.org"
remote_tile_dir = "/var/www/tiles/baltic_2026"
remote_html_dir = "/var/www/html/baltic"
remote_url = "https://tiles.oceanserver.org"
```
