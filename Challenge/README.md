# Ops Data Challenge: managing a renewables portfolio

One of FlexPower's main products is the so-called **RouteToMarket**: bringing the energy produced by renewables' 
assets to the european electricity markets.

FlexPower signs contracts with renewable assets' owners, and then sells the energy produced by these assets on the market, hopefully in a profitable way thanks to its trading expertise. 

At the end of each month, FlexPower invoices the asset owners based on the production of their assets: 
FlexPower pays the asset owner an agreed upon price for each produced mwh and the asset owner pays FlexPower 
a fee for the services provided, per mwh.\
There are different pricing and fee models, i.e. ways to compute the price and fee, to accommodate different types of assets and customers.\

In this challenge, we will look at different aspects of marketing renewables and present different data entities involved.\
The goal will be to compute different financial flows and present them in a way that helps stakeholders understand the performance of the portfolio and of single assets.

## Assets "base" data
With **base data** we usually mean static attributes of assets. These can be technical, contractual or financial.

The assets' base data is stored as a CSV file, `assets_base_data.csv`. It contains the following columns:
- **asset_id**: a unique alphanumerical identifier for the asset.
- **metering_point_id**: a unique alphanumerical identifier for the asset's metering point.
- **capacity**: the installed capacity of the asset, in MW.
- **technology**: the technology used by the asset, can be either "solar" or "wind".
- **contract_begin**: the date when the asset's contract with FlexPower began.
- **contract_end**: the date when the asset's contract with FlexPower ends, can be none.
- **price_model**: the pricing model used by the asset, can be either "**fixed**" or "**market**".
- **price**: the price paid by FlexPower to the asset owner, in €/MWh, if asset in the "fixed" price model.
- **fee_model**: the fee model used by the asset, can be either "**fixed_as_produced**", "**fixed_for_capacity**" or "**percent_of_market**".
- **fee**: the fee paid by the asset owner to FlexPower, in €/MWh, if asset in the "fixed" fee model.
- **fee_percent**: percent of the market value paid by the asset owner to FlexPower per mwh, in %, if asset in the "percent_of_market" fee model.

##  Forecasts and measured production

### Forecasts:

For each asset, FlexPower forecasts the electricity production. 
This forecast allows FlexPower to know how much electricity will be available to sell on the markets for a given hour in the future.

This forecast is updated every hour and is stored as a json. The production is forecasted for the next 24 hours with an hourly resolution.\ 
Each point represents the average power produced by the asset over a given hour in the future.
The values are in kilowatt and the format is as follows:

```json
{
  "asset_id": "asset_123",
  "version": "2024-10-13T12:00:00Z",
  "values": {
    "2024-10-13T13:00:00Z": 100,
    "2024-10-13T14:00:00Z": 110
  }
}
```

### Measured production:
Each asset is connected to the electricity grid through a so-called "metering point". 
The production measured at this point is communicated to FlexPower by the Distribution System Operator (DSO) every day for the previous day. 

The values here are used to invoice the asset owner and we get data for all assets at once, in a csv file. The unit is kilowatt.


## Energy Trading in a Nutshell

Energy trading happens in an **exchange**, a market where traders working for energy producers 
(solar plants, nuclear power plants, ...) and consumers (B2C energy providers, big 
energy consuming industries like steel and trains...) submit orders to buy and sell energy.
One of the major exchanges in Europe is called [**EPEX**](https://en.wikipedia.org/wiki/European_Power_Exchange).

These orders consist of a volume (in Megawatt, rounded to one decimal) over a predefined period of
time, called **delivery period** (for example between 12:00, the **delivery start** and 13:00, 
the **delivery end**, on a given day) and for a given price per megawatt hour (referred to as mwh).

If the prices of two orders with opposite sides match, i.e the buy price is higher than the sell price, 
then, a trade is generated. 
For example if the orderbook contains an order to sell 10 mw for 10 euros/mwh and another trader 
submits an order to buy 5 mw for 11 euros/mwh, the orders are matched by the exchange and a trade is 
generated for 5mw at 10 euros/mwh.

The trading revenues are calculated by multiplying the volume with the price over all trades: for a buy trade, the revenue is 
negative, for a sell deal, the revenue is positive.

A list of trades that our trading floor executed are provided as a json. Volumes are in megawatt and prices are in euro per megawatt-hour.

We also provide and index price for the market, in a csv file. The index price is the volume weighted average price of all trades executed on the market, in euro per megawatt-hour.

## Invoicing
In the invoices we send out to the customers, we need to compute the following entries:
* infeed payout: multiplying the production of the asset by the price we agreed upon with the asset owner, depending on the price model.
    * For the "fixed" pricing model, we just multiply the total produced volume with the price from the base data. This is a more "conservative" pricing model.
    * For the "market" pricing model, we multiply the production of the asset, hour by hour, with the market index price. This is a price model that is more sensitive to the market price fluctuations and can yield a better payout if the asset is producing at times where the price on the market is high.

* fees: multiplying the production of the asset by the fee we agreed upon with the asset owner, depending on the fee model.
    * For the "fixed_as_produced" fee model, we multiply the total produced volume with the fee from the base data.
    * For the "fixed_for_capacity" fee model, we multiply the installed capacity of the asset with the fee from the base data.
    * For the "percent_of_market" fee model, we multiply the total produced volume of the asset with the average of the market index price and the fee percent from the base data.

For each of these entries we also compute the unitary net amount, the VAT (at with a rate of 19%) and the total gross amount.

## Imbalance

The electricity grid functions properly only when the production and consumption are balanced at all times and there are financial incentives to achieve that.

If our forecasted production is different from the measured production (higher or lower), we have an imbalance volume and for that we have to pay a penalty.\
The imbalance penalty is provided as a csv file and the unit is euro per megawatt-hour.
We can then compute the imbalance cost for each asset in our portfolio.

## The challenge

Your goal is to help FlexPower make sense of all this data, in particular:
- compute the trading revenues.
- compute the imbalance cost for each asset.
- compute invoices for each asset.
- create a report that helps FlexPower understand the performance of its portfolio and each single asset.

You can use any tools you want, but we recommend using python for any coding involved and SQL for queries and data processing.\
Feel free to add any indicators and/or plots that you feel would help presenting these results to stakeholders and 
report any insights that might emerge from the data. 


