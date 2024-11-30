import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / ".env")

try:
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
except:
    raise Exception("Create a .env file on the project root with the HOPSWORKS_API_KEY")

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "daily_demand_feature_group"
FEATURE_GROUP_VERSION = 1

# BAs for which we also have map layers: 53 in total.
BAS = [
    "AZPS",
    "AECI",
    "BPAT",
    "CISO",
    "CPLE",
    "CHPD",
    "DOPD",
    "DUK",
    "EPE",
    "ERCO",
    "FPL",
    "FPC",
    "GVL",
    "HST",
    "IPCO",
    "IID",
    "JEA",
    "LDWP",
    "LGEE",
    "NWMT",
    "NEVP",
    "ISNE",
    "NYIS",
    "PACW",
    "PACE",
    "FMPP",
    "GCPD",
    "PJM",
    "PGE",
    "PSCO",
    "PNM",
    "PSEI",
    "BANC",
    "SRP",
    "SCL",
    "SCEG",
    "SC",
    "SPA",
    "SOCO",
    "TPWR",
    "TAL",
    "TEC",
    "TVA",
    "TIDC",
    "WAUW",
    "AVA",
    "SEC",
    "TEPC",
    "WALC",
    "WACM",
    "MISO",
    "CPLW",
    "SWPP",
]
