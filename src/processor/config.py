
import pathlib

CRUISENAME = "bioreactors1"
BASE_TILE_DIR = "/data/slow/web_share/cruise/tiles/"
DATA_DIR = "/data/raid/cruise_support/"
REMOTE_HTML_DIR = "/var/www/html/cruise/"

def settings():
    return dict(
        cruise_name = CRUISENAME,
        lat1 = -55,
        lat2 = -10,
        lon1 = -75,
        lon2 = -5,
        tile_dir = BASE_TILE_DIR + CRUISENAME,
        remote_html_dir = REMOTE_HTML_DIR
        remote_tile_dir = REMOTE_HTML_DIR + "/tiles/" + CRUISENAME
        data_dir = DATA_DIR + CRUISENAME,
        zoom_levels = [0,1,2,3,4,5,6,7,8,9]
)
