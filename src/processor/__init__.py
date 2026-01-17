
import pandas as pd

from . import config, tile, layer_config
settings = config.settings.from_env(ENV)
#def day(dtm):
#    processor_day.day(dtm)

class DateInFutureError(Exception):
    pass

def day(dtm):
    pass

def today():
    dtm = pd.Timestamp.now().normalize()
    tile.all(dtm)
    if config.settings.get("remote_sync"):
        tile.sync()

def yesterday():
    dtm = pd.Timestamp.now().normalize()-pd.Timedelta(1,"D")
    tile.all(dtm)
    if config.settings.get("remote_sync"):
        print("Sync tiles")
        tile.sync()
        layer_config.sync()

def last_days(days=7):
    dtm1 = pd.Timestamp.now().normalize()-pd.Timedelta(days,"D")
    dtm2 = pd.Timestamp.now().normalize()
    for dtm in pd.date_range(dtm1, dtm2):
        tile.all(dtm)
    if config.settings.get("remote_sync"):
        tile.sync()
        layer_config.sync()
