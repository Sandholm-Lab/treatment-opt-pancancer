"""
Grouping of cell lines by type of tissue.
"""

BREAST = [
    "AU565",
    "BT20",
    "BT474",
    "BT549",
    "CAL851",
    "CAMA1",
    "EFM19",
    "HCC1187",
    "HCC1395",
    "HCC1569",
    "HCC1806",
    "HCC70",
    "HMC18",
    "MDAMB415",
    "MDAMB436",
    "MDAMB468",
    "SKBR3",
    "UACC812",
    "ZR751",
    "ZR7530",
]

LARGE_INTESTINE = [
    "COLO201",
    "COLO320",
    "HCC56",
    "LS123",
    "LS411N",
    "NCIH747",
    "SKCO1",
    "SNUC2A",
    "SW1417",
    "SW403",
    "SW480",
    "T84",
]

LUNG = [
    "CAL12T",
    "DV90",
    "EBC1",
    "HCC2935",
    "HCC4006",
    "HCC78",
    "HCC827",
    "LCLC103H",
    "LOUNH91",
    "LU99",
    "LUDLU1",
    "NCIH1048",
    "NCIH1092",
    "NCIH1184",
    "NCIH1299",
    "NCIH1341",
    "NCIH1355",
    "NCIH1563",
    "NCIH1573",
    "NCIH1648",
    "NCIH1650",
    "NCIH1651",
    "NCIH1666",
    "NCIH1693",
    "NCIH1694",
    "NCIH1703",
    "NCIH1792",
    "NCIH1793",
    "NCIH1869",
    "NCIH1915",
    "NCIH1944",
    "NCIH1975",
    "NCIH2023",
    "NCIH2030",
    "NCIH2087",
    "NCIH211",
    "NCIH2170",
    "NCIH2172",
    "NCIH2228",
    "NCIH2286",
    "NCIH23",
    "NCIH2444",
    "NCIH3255",
    "NCIH358",
    "NCIH441",
    "NCIH520",
    "NCIH522",
    "NCIH647",
    "NCIH650",
    "NCIH661",
    "NCIH727",
    "NCIH810",
    "PC14",
    "RERFLCAI",
    "RERFLCMS",
    "SBC5",
    "SHP77",
    "SKLU1",
    "SKMES1",
    "SW1271",
    "SW1573",
    "SW900",
    "T3M10",    
]

PANCREAS = [
    "HPAFII",
    "HUPT3",
    "KP2",
    "KP3",
    "KP4",
    "L33",
    "PK45H",
    "PK59",
    "SU8686",
    "SW1990",
]

SKIN = [
    "A2058",
    "C32",
    "COLO679",
    "HMCB",
    "HT144",
    "K029AX",
    "MDAMB435S",
    "MELHO",
    "RPMI7951",
    "SKMEL24",
    "SKMEL30",
    "UACC257",
    "UACC62",
    "WM115",
    "WM1799",
    "WM2664",
    "WM793",
    "WM88",
    "WM983B",
]

INITIAL_LINES = [
    'DV90',
    'HS695T',
    'NCIH1092',
    'PK59',
    'A2058',
    'SKMEL24', 
    'SKMEL30',
]

# Subsets for experiments
# -------------------------------------------------------------------------------------------------
# for details se experimental list


KB = [
    'LS123',
    'SW1417'
]

SK = [
    "SKMEL24",
    "SKMEL30"
]

WM = [
    "WM115",
    "WM1799",
    "WM2664",
    "WM793",
    "WM88",
    "WM983B",
]

WME = [
    "WM1799",
    "WM793",
    "WM88",
    "WM983B",
]

COLO = [
    "COLO201",
    "COLO320",
    "COLO679",
]


# GROUPS FOR EXPERIMENTS
# -------------------------------------------------------------------------------------------------

GA2058 = [
    'A2058',
    'MDAMB435S',
    'K029AX'
]

SW = [
    "SW403",
    "SW480",    
]

WMD = [
    "WM115",
    "WM2664",
]


GA2058_SW = GA2058 + SW
GA2058_WMD = GA2058 + WMD
SW_WMD = SW + WMD
GA2058_SW_WMD = GA2058 + SW + WMD

# -------------------------------------------------------------------------------------------------

LINES = {
    "breast": BREAST,
    "intestine": LARGE_INTESTINE,
    "lung": LUNG,
    "pancreas": PANCREAS,
    "skin": SKIN,
    "initial": INITIAL_LINES,
    "ga2058": GA2058,
    "kb": KB,
    "sk": SK,
    "wm": WM,
    "wmd": WMD,
    "wme": WME,
    "sw": SW,
    "colo": COLO,
    "ga2058_sw": GA2058_SW,
    "ga2058_wmd": GA2058_WMD,
    "sw_wmd": SW_WMD,
    "ga2058_sw_wmd": GA2058_SW_WMD,
}

def retrieve_lines(line):
    if line in LINES:
        return LINES[line]
    else:
        raise ValueError("Specified tissue is unknown.")
