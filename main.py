import numpy as np
import pandas as pd
import matplotlib.dates as md
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
import seaborn as sb
import os
import json
from datetime import datetime
from pathlib import Path

### Input Data ####################################################################################################

base_path = Path(__file__).parent
file_path1 = base_path / "Challenge" / "forecasts" 
file_path2 = base_path / "Challenge"

assets_forecast = {}

for i in range(1, 5): # assets forcast in kW

    with open(f"{file_path1}/a{i}.json", "r") as f:
        data = json.load(f)
        s = pd.Series(data["values"])
        s.index = pd.to_datetime(s.index, utc=True)
        assets_forecast[f"a_{i}"] = s

with open(f'{file_path2}/trades.json', "r") as f:
    trades = json.load(f) # trades

trades = pd.DataFrame(trades)
trades["delivery_start"] = pd.to_datetime(trades["delivery_start"], utc=True)
trades["delivery_end"] = pd.to_datetime(trades["delivery_end"], utc=True)

base_data = pd.read_csv(f'{file_path2}/assets_base_data.csv', sep=';', index_col=0) # asset base data
penalty = pd.read_csv(f'{file_path2}/imbalance_penalty.csv', sep=';', index_col=0) # imbalance penalty
penalty.index = pd.to_datetime(penalty.index, utc=True)
market_price = pd.read_csv(f'{file_path2}/market_index_price.csv', sep=';', index_col=0) # market index price
market_price.index = pd.to_datetime(market_price.index, utc=True)
measured = pd.read_csv(f'{file_path2}/measured_20241013.csv', sep=';', index_col=0) # measured production in [kW]
measured.index = pd.to_datetime(measured.index, utc=True)

### Invoicing ####################################################################################################

invoicing = []

for asset_id, row in base_data.iterrows():
        
    mp_id = row["metering_point_id"]
    price_model = row["price_model"]
    price = row["price__eur_per_mwh"]
    fee_model = row["fee_model"]
    fee = row["fee__eur_per_mwh"]
    capacity = row["capacity__kw"] # Capacity is in MW
    fee_percent = row["fee_percent"] / 100 # in %

    if mp_id in measured.columns:

        total_production_mwh = measured[mp_id].sum() * 0.001 # in kWh

        if row["price_model"] == "fixed":
            payout = total_production_mwh * price # in €

        elif row["price_model"] == "market": 
            
            production_mwh = measured[mp_id] * 0.001

            if market_price['market_index_price'].index[0] == production_mwh.index[0]:
                payout = market_price['market_index_price'] * production_mwh
                payout = payout.sum()
        
        else: 
        
            payout = None

        if row["fee_model"] == "fixed_as_produced":
            asset_fee = total_production_mwh * fee

        elif row["fee_model"] == "fixed_for_capacity":
            
            asset_fee = capacity * fee
        
        elif row["fee_model"] == "percent_of_market":

            asset_fee = total_production_mwh * (market_price["market_index_price"].sum()/len(market_price)) * fee_percent

        else:

            asset_fee = None

        invoicing.append({
                "asset_id": asset_id,
                "metering_point_id": mp_id,
                "total_production_mwh": total_production_mwh,
                "price_mwh": price,
                "infeed_payout_netto": round(payout,2),
                "infeed_payout_brutto": round(payout * 1.19,2),
                "fee_netto": round(asset_fee,2),
                "fee_brutto": round(asset_fee * 1.19,2)
            })
        
invoicing = pd.DataFrame(invoicing)

### Imbalance ########################################################################################

imbalance = []

for asset_id, row in base_data.iterrows():

    mp_id = row["metering_point_id"]

    forcast = assets_forecast[asset_id] 
    exact = measured[mp_id]

    diff = (forcast - exact).abs() # Difference in KW
    diff_mwh = diff * 0.001 # Difference in MW

    cost = diff_mwh * penalty["imbalance_penalty"]
    sum_cost = cost.sum() # in €

    imbalance.append({
                "asset_id": asset_id,
                "metering_point_id": mp_id,
                "penalty_cost": round(sum_cost,2)
            })

imbalance = pd.DataFrame(imbalance)

### Trade Revenue ###################################################################################

trade_revenue = []

for nr, row in trades.iterrows():

    if row["side"] == "buy":
        rev = -row["quantity"] * row["price"]
    elif row["side"] == "sell":
        rev = row["quantity"] * row["price"]

    for a in market_price.iterrows():
        
        if row["delivery_start"] == a[0]:
            market_pr = a[1]["market_index_price"]

            if row["side"] == "buy":
                value = -row["quantity"] * market_pr
            elif row["side"] == "sell":
                value = row["quantity"] * market_pr

    trade_revenue.append({
                    "trade_id": row["trade_id"],
                    "price": row["price"],
                    "revenue in €": round(rev,2),
                    "market_price": market_pr,
                    "market_value in €": round(value,2)
                 })  

trade_revenue = pd.DataFrame(trade_revenue)

    