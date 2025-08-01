import numpy as np
import pandas as pd
import matplotlib.dates as md
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
import seaborn as sb
import os
import json
import datetime as dt
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

assets_forecast = pd.DataFrame(assets_forecast)        

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
infeed = []

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

            payout = (measured[mp_id] * 0.001) * price # in €
            payout_VAT = payout * 1.19


        elif row["price_model"] == "market": 
            
            production_mwh = measured[mp_id] * 0.001

            if market_price['market_index_price'].index[0] == production_mwh.index[0]:
            
                payout = market_price['market_index_price'] * (measured[mp_id] * 0.001)
                payout_VAT = payout * 1.19
    
        else: 
        
            payout = None

        if row["fee_model"] == "fixed_as_produced":
            asset_fee = total_production_mwh * fee
           

        elif row["fee_model"] == "fixed_for_capacity":
            
            asset_fee = (capacity * 0.001) * fee
        
        elif row["fee_model"] == "percent_of_market":

            asset_fee = total_production_mwh * (market_price["market_index_price"].sum()/len(market_price)) * fee_percent

        else:

            asset_fee = None

        infeed.append({
                    f"{mp_id}": round(payout,2)
                })        
        infeed.append({
                    f"{mp_id}_VAT": round(payout_VAT,2)
                })

        invoicing.append({
                "asset_id": asset_id,
                "metering_point_id": mp_id,
                "total_production_mwh": total_production_mwh,
                "price_mwh": price,

                "fee_netto": round(asset_fee,2),
                "fee_brutto": round(asset_fee * 1.19,2)
            })
        
infeed_payout = pd.concat([v for d in infeed for v in d.values()], axis=1)
infeed_payout.columns = [list(d.keys())[0] for d in infeed] 
        
invoicing = pd.DataFrame(invoicing)

### Imbalance ####################################################################################################

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

### Trade Revenue ####################################################################################################

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


### Asset Revenue ####################################################################################################

asset_revenue = []


for time, asset in assets_forecast.iterrows():

        for nr, row in trades.iterrows():

            if row["delivery_start"] == time:

                a1_rev = (asset["a_1"] * 0.001) * row["price"]
                a2_rev = (asset["a_2"] * 0.001) * row["price"]
                a3_rev = (asset["a_3"] * 0.001) * row["price"]
                a4_rev = (asset["a_4"] * 0.001) * row["price"]
           
                asset_revenue.append({
                            "trade_id": row["trade_id"],
                            "price": row["price"],
                            "a1_rev": round(a1_rev,2),
                            "a2_rev": round(a2_rev,2),
                            "a3_rev": round(a3_rev,2),
                            "a4_rev": round(a4_rev,2)
                        })     

            
asset_revenue = pd.DataFrame(asset_revenue)

### Plotten #############################################################################################

asset_ids = ["asset 1", "asset 2", "asset 3", "asset 4"]

# Profits assets
profit_a1 = asset_revenue["a1_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_1", "fee_brutto"].values[0] - infeed_payout["mp_1_VAT"].sum() - imbalance.loc[imbalance["asset_id"] == "a_1", "penalty_cost"].values[0]
profit_a2 = asset_revenue["a2_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_2", "fee_brutto"].values[0] - infeed_payout["mp_2_VAT"].sum() - imbalance.loc[imbalance["asset_id"] == "a_2", "penalty_cost"].values[0]
profit_a3 = asset_revenue["a3_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_3", "fee_brutto"].values[0] - infeed_payout["mp_3_VAT"].sum() - imbalance.loc[imbalance["asset_id"] == "a_3", "penalty_cost"].values[0]
profit_a4 = asset_revenue["a4_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_4", "fee_brutto"].values[0] - infeed_payout["mp_4_VAT"].sum() - imbalance.loc[imbalance["asset_id"] == "a_4", "penalty_cost"].values[0]
total_profit = profit_a1 + profit_a2 + profit_a3 + profit_a4

# Revenue assets
rev_a1 = asset_revenue["a1_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_1", "fee_brutto"].values[0] 
rev_a2 = asset_revenue["a2_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_2", "fee_brutto"].values[0] 
rev_a3 = asset_revenue["a3_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_3", "fee_brutto"].values[0] 
rev_a4 = asset_revenue["a4_rev"].sum() + invoicing.loc[invoicing["asset_id"] == "a_4", "fee_brutto"].values[0] 
total_rev = rev_a1 + rev_a2 + rev_a3 + rev_a4

# Costs assets
costs_a1 = infeed_payout["mp_1_VAT"].sum() + imbalance.loc[imbalance["asset_id"] == "a_1", "penalty_cost"].values[0]
costs_a2 = infeed_payout["mp_2_VAT"].sum() + imbalance.loc[imbalance["asset_id"] == "a_2", "penalty_cost"].values[0]
costs_a3 = infeed_payout["mp_3_VAT"].sum() + imbalance.loc[imbalance["asset_id"] == "a_3", "penalty_cost"].values[0]
costs_a4 = infeed_payout["mp_4_VAT"].sum() + imbalance.loc[imbalance["asset_id"] == "a_4", "penalty_cost"].values[0]
total_costs = costs_a1 + costs_a2 + costs_a3 + costs_a4

# return
return_ = (total_profit/total_costs) * 100

# time
date = "13.10.2024 21:00"

# ───── Dashboard ──────────────────────────────────────────────────────
fig = pl.figure(figsize=(18, 10))
main_gs = gridspec.GridSpec(7, 2, height_ratios=[0.01, 0.05, 0.05, 1, 1, 0.05, 1], hspace=0.6, wspace=0.25)

# ─── Titel ─────────────────────────────────────────────────────────────
ax_title = pl.subplot(main_gs[0, :])
ax_title.set_axis_off()
ax_title.text(0.5, 0.5, "FlexPower Asset Performance Dashboard", ha="center", va="center", fontsize=18, fontweight="bold") 

# ─── KPI- ─────────────────────────────────────────────────────────
ax_kpi = pl.subplot(main_gs[1, :])
ax_kpi.set_axis_off()
kpi_text = (
    f"Total Revenue: {total_rev:,.2f} €   | "
    f"Total Costs: {total_costs:,.2f} €   | "
    f"Total Profit: {total_profit:,.2f} €   | "
    f"Return: {return_:.2f} %\n"
    f"Date: {date}   | "
    f"Time Horizon: 24 hours"
)

ax_kpi = pl.subplot(main_gs[1, :])
ax_kpi.set_axis_off()
ax_kpi.text(0.5, 0.5, kpi_text, ha="center", va="center", fontsize=13,  fontweight="bold", linespacing=1.5 )

pos = ax_kpi.get_position()
new_pos = [pos.x0, pos.y0 - 0.02, pos.width, pos.height]
ax_kpi.set_position(new_pos)

# ─── Seperation ──────────────────────────────────────────────

ax_line = pl.subplot(main_gs[2, :])
ax_line.axhline(0, color='black', linewidth=3)
ax_line.set_axis_off()

# ─── Plot Assets ────────────────────────────────────────────

# Asset 1 #########
gs_asset1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[3, 0], width_ratios=[4, 1], wspace=0.3)

# left 
ax_bar1 = pl.subplot(gs_asset1[0])
ax_bar1.bar("Revenue (€)", asset_revenue["a1_rev"].sum(), color="green", label="Revenue", width=0.6)
ax_bar1.bar("Revenue (€)", invoicing.loc[invoicing["asset_id"] == "a_1", "fee_brutto"].values[0], bottom=asset_revenue["a1_rev"].sum(), color="blue", label="Fee", width=0.6)
ax_bar1.bar("Cost (€)", -infeed_payout["mp_1_VAT"].sum(), color="red", label="Infeed", width=0.6)
ax_bar1.bar("Cost (€)", -imbalance.loc[imbalance["asset_id"] == "a_1", "penalty_cost"].values[0], bottom=-infeed_payout["mp_1_VAT"].sum(), color="orange", label="Imbalance", width=0.6)
ax_bar1.set_title(f"{asset_ids[0]}", fontsize=12, fontweight="bold")
ax_bar1.axhline(0, color="black", linewidth=0.8)
ax_bar1.legend(fontsize=10, loc="upper right")

# right
ax_text1 = pl.subplot(gs_asset1[1])
ax_text1.set_axis_off()
ax_text1.text(0.5, 0.5, f"Profit:\n{profit_a1:.2f} €", ha="right", va="center",
              fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0"))

# Asset 2 #########
gs_asset2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[3, 1], width_ratios=[4, 1], wspace=0.3)

# left 
ax_bar2 = pl.subplot(gs_asset2[0])
ax_bar2.bar("Revenue (€)", asset_revenue["a2_rev"].sum(), color="green", label="Revenue", width=0.6)
ax_bar2.bar("Revenue (€)", invoicing.loc[invoicing["asset_id"] == "a_2", "fee_brutto"].values[0], bottom=asset_revenue["a2_rev"].sum(), color="blue", label="Fee", width=0.6)
ax_bar2.bar("Cost (€)", -infeed_payout["mp_2_VAT"].sum(), color="red", label="Infeed", width=0.6)
ax_bar2.bar("Cost (€)", -imbalance.loc[imbalance["asset_id"] == "a_2", "penalty_cost"].values[0], bottom=-infeed_payout["mp_2_VAT"].sum(), color="orange", label="Imbalance", width=0.6)
ax_bar2.set_title(f"{asset_ids[1]}", fontsize=12, fontweight="bold")
ax_bar2.axhline(0, color="black", linewidth=0.8)
ax_bar2.legend(fontsize=10, loc="upper right")

# right
ax_text2 = pl.subplot(gs_asset2[1])
ax_text2.set_axis_off()
ax_text2.text(0.5, 0.5, f"Profit:\n{profit_a2:.2f} €", ha="right", va="center",
              fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0"))

# Asset 3 #########
gs_asset3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[4, 0], width_ratios=[4, 1], wspace=0.3)

# left 
ax_bar3 = pl.subplot(gs_asset3[0])
ax_bar3.bar("Revenue (€)", asset_revenue["a3_rev"].sum(), color="green", label="Revenue", width=0.6)
ax_bar3.bar("Revenue (€)", invoicing.loc[invoicing["asset_id"] == "a_3", "fee_brutto"].values[0], bottom=asset_revenue["a3_rev"].sum(), color="blue", label="Fee", width=0.6)
ax_bar3.bar("Cost (€)", -infeed_payout["mp_3_VAT"].sum(), color="red", label="Infeed", width=0.6)
ax_bar3.bar("Cost (€)", -imbalance.loc[imbalance["asset_id"] == "a_3", "penalty_cost"].values[0], bottom=-infeed_payout["mp_3_VAT"].sum(), color="orange", label="Imbalance", width=0.6)
ax_bar3.set_title(f"{asset_ids[2]}", fontsize=12, fontweight="bold")
ax_bar3.axhline(0, color="black", linewidth=0.8)
ax_bar3.legend(fontsize=10, loc="upper right")

# right
ax_text3 = pl.subplot(gs_asset3[1])
ax_text3.set_axis_off()
ax_text3.text(0.5, 0.5, f"Profit:\n{profit_a3:.2f} €", ha="right", va="center",
              fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0"))

# Asset 4 #########
gs_asset4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[4, 1], width_ratios=[4, 1], wspace=0.3)

# left 
ax_bar4 = pl.subplot(gs_asset4[0])
ax_bar4.bar("Revenue (€)", asset_revenue["a4_rev"].sum(), color="green", label="Revenue", width=0.6)
ax_bar4.bar("Revenue (€)", invoicing.loc[invoicing["asset_id"] == "a_4", "fee_brutto"].values[0], bottom=asset_revenue["a4_rev"].sum(), color="blue", label="Fee", width=0.6)
ax_bar4.bar("Cost (€)", -infeed_payout["mp_4_VAT"].sum(), color="red", label="Infeed", width=0.6)
ax_bar4.bar("Cost (€)", -imbalance.loc[imbalance["asset_id"] == "a_4", "penalty_cost"].values[0], bottom=-infeed_payout["mp_4_VAT"].sum(), color="orange", label="Imbalance", width=0.6)
ax_bar4.set_title(f"{asset_ids[3]}", fontsize=12, fontweight="bold")
ax_bar4.axhline(0, color="black", linewidth=0.8)
ax_bar4.legend(fontsize=10, loc="upper right")

# right
ax_text4 = pl.subplot(gs_asset4[1])
ax_text4.set_axis_off()
ax_text4.text(0.5, 0.5, f"Profit:\n{profit_a4:.2f} €", ha="right", va="center",
              fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0"))

# ─── Seperation ──────────────────────────────────────────────

ax_line = pl.subplot(main_gs[5, :])
ax_line.axhline(0, color='black', linewidth=3)
ax_line.set_axis_off()

# ─── Comparison Plots ──────────────────────────────────────────────
# Solar
ax_comp1 = pl.subplot(main_gs[6,0])

ax_comp1.set_title("Profit / Capacity Solar", fontsize=12, fontweight="bold")
ax_comp1.set_ylabel("Profit per KW (€/KW)", fontsize=12)
ax_comp1.bar("asset 1", profit_a1/base_data.loc["a_1", "capacity__kw"], color="steelblue", width=0.5)
ax_comp1.bar("asset 2", profit_a2/base_data.loc["a_2", "capacity__kw"], color="steelblue",width=0.5)

ax_comp1.bar("asset 4", profit_a4/base_data.loc["a_4", "capacity__kw"], color="steelblue",width=0.5)

pos = ax_comp1.get_position()  
new_width = pos.width * 0.8 
new_left = pos.x0 + (pos.width - new_width) / 2  
ax_comp1.set_position([new_left, pos.y0, new_width, pos.height])

# Wind
ax_comp2 = pl.subplot(main_gs[6,1])

ax_comp2.set_title("Profit / Capacity Wind", fontsize=12, fontweight="bold")
ax_comp2.set_ylabel("Profit / kW (€/kW)", fontsize=12)
ax_comp2.bar(" ", 0, color="steelblue", width=0.4)
ax_comp2.bar("asset 3", profit_a3/base_data.loc["a_3", "capacity__kw"], color="steelblue", width=0.5)
ax_comp2.bar("  ", 0, color="steelblue", width=0.4)

pos = ax_comp2.get_position() 
new_width = pos.width * 0.8    
new_left = pos.x0 + (pos.width - new_width) / 2  

ax_comp2.set_position([new_left, pos.y0, new_width, pos.height])

# ─── show ───────────────────────────────────────────────────────────
output_path = "Output/"

sb.despine()
sb.set_palette("bright")
fig.subplots_adjust(
    top=0.9,    
    bottom=0.1, 
    left=0.1,   
    right=0.9   
)

fig.savefig(f'{output_path}Dashboard.pdf', bbox_inches="tight", format="pdf")
pl.show(block=False)
