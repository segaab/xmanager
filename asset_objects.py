# asset_objects.py
from dataclasses import dataclass
from typing import List

@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

assets_list: List[Asset] = [
    Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol="GC=F"),
    Asset(name="Silver", cot_name="SILVER - COMMODITY EXCHANGE INC.", symbol="SI=F"),
    Asset(name="Euro FX", cot_name="EURO FX - CHICAGO MERCANTILE EXCHANGE", symbol="6E=F"),
    Asset(name="Japanese Yen", cot_name="JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", symbol="6J=F"),
    Asset(name="British Pound", cot_name="BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", symbol="6B=F"),
    Asset(name="Canadian Dollar", cot_name="CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", symbol="6C=F"),
    Asset(name="Australian Dollar", cot_name="AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", symbol="6A=F"),
    Asset(name="Swiss Franc", cot_name="SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", symbol="6S=F"),
    Asset(name="S&P 500 E-Mini", cot_name="S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", symbol="ES=F"),
    Asset(name="NASDAQ-100 E-Mini", cot_name="NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", symbol="NQ=F"),
    Asset(name="Dow Jones E-Mini", cot_name="DOW JONES INDUSTRIAL AVERAGE - CHICAGO MERCANTILE EXCHANGE", symbol="YM=F"),
    Asset(name="Crude Oil", cot_name="CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE", symbol="CL=F"),
    Asset(name="Natural Gas", cot_name="NATURAL GAS - NEW YORK MERCANTILE EXCHANGE", symbol="NG=F"),
    Asset(name="Copper", cot_name="COPPER - COMMODITY EXCHANGE INC.", symbol="HG=F"),
    Asset(name="Platinum", cot_name="PLATINUM - NEW YORK MERCANTILE EXCHANGE", symbol="PL=F"),
    Asset(name="Palladium", cot_name="PALLADIUM - NEW YORK MERCANTILE EXCHANGE", symbol="PA=F"),
]