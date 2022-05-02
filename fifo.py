import json
from datetime import datetime
from dateutil.parser import parse
import pandas as pd


def load_file(filename=None, **kwargs):
    return pd.read_csv(filename, **kwargs)

def prepare(df=None):
    """Creates operation column (buy or sell), casts types and drops unused cols.
    Returns DF with refid as index
    """
    df["operation"] = df["amount"].apply(lambda x: "sell" if x < 0 else "buy")
    df["time"] = pd.to_datetime(df["time"])
    for col in ["txid", "refid", "type", "subtype", "asset", "operation"]:
        df[col] = df[col].astype("string")
    df.drop(columns=["subtype", "aclass", "balance"], inplace=True)
    df.set_index("refid", inplace=True)
    return df

def split_types(df=None):
    """Splits data frame on transactions, changes (trades), stakings and deposits.
    """
    transactions = df[df["txid"].notnull()]
    changes = transactions[
        (transactions["type"] != "transfer") &
        (transactions["type"] != "staking") &
        (transactions["type"] != "deposit")
    ]
    staks = transactions[
        (transactions["type"] == "transfer") |
        (transactions["type"] == "staking")
    ]
    deposits = transactions[transactions["type"] == "deposit"]
    return transactions, changes, staks, deposits

def agg_transactions(df=None):
    """Self joins to link assets sells with buys.
    """
    sells = df[df["operation"] == "sell"]
    buys = df[df["operation"] == "buy"]
    joined = buys.join(
        sells,
        on="refid",
        how="left",
        lsuffix="_buy",
        rsuffix="_sell"
    )
    return joined.drop(
        columns=[
            "type_buy",
            "type_sell",
            "txid_buy",
            "txid_sell",
            "time_sell",
            "operation_buy",
            "operation_sell"
        ]
    ).rename(columns={"time_buy": "time"})






    
def fill_with(amountt=None, options=None):
    assignment = []
    quantity_to_fill = amountt
    for idx, item in options.iterrows():
        if item["amount_buy"] > quantity_to_fill:
            assignment.append({"idx": idx, "quantity": quantity_to_fill})
            quantity_to_fill = 0
        else:
            assignment.append({"idx": idx,  "quantity": item["amount_buy"]})
            quantity_to_fill -= item["amount_buy"]
        if quantity_to_fill == 0:
            break
    return list(filter(lambda x: abs(x["quantity"]) > 0, assignment))

def balances(data=None, pair=None):
    """Returns a df with refid as key and columns for each refid which the founds come from
    also with quantity.
    """
    sells = data[data["asset_sell"] == pair]
    buys = data[data["asset_buy"] == pair]
    df_refs = []
    for idx, item in sells.iterrows():
        amount = item["amount_sell"]
        origin = buys[buys["time"] < item["time"]].sort_values(by="time", ascending=True)
        computed_sells = fill_with(abs(amount), origin)
        if computed_sells == []:
            print("ERROR, sells not linked to foundings")
            continue
        for item_dist in computed_sells:
            index, diff = item_dist["idx"], item_dist["quantity"]
            buys.at[index, "amount_buy"] -= diff
            item_dist["refid"] = idx
        df_ref = pd.DataFrame(computed_sells)
        df_ref["idx"] = df_ref["idx"].astype("string")
        df_refs.append(df_ref)
    if df_refs == []:
        return None
    return pd.concat(df_refs)

def attach_buy_prices(data=None, asset_founding=None):
    """Data comes with refid index"""
    if asset_founding is None:
        return None
    asset_indexed = asset_founding.reset_index() \
        .rename(columns={"refid": "refid_origin", "idx": "refid"})
    asset_indexed = asset_indexed.set_index("refid")
    data_ = data[["price_EUR_buy"]]
    joined = asset_indexed.join(data_, on="refid", how="left") \
        .reset_index() \
        .rename(
            columns={
                "refid": "refid_foundings",
                "refid_origin": "refid",
                "price_EUR_buy": "price_bought_EUR"}
        ) \
        .set_index("refid") \
        .drop(columns="index")
    return joined

def list_assets(data=None):
    assets_l = data["asset_buy"]
    assets_r = data["asset_sell"]
    assets = assets_l.values.tolist() + assets_r.values.tolist()
    return list(set(assets))


def build_assets_prices(assets_dict=None):
    dfs = []
    for key, value in assets_dict.items():
        if value is not None:
            df = load_file(
                value,
                names=["timestamp", "open", "high", "low", "close", "volume", "trades"]
            )
            df["asset"] = key
            df["asset"] = df["asset"].astype("string")
            dfs.append(df)
    return pd.concat(dfs)

def attach_prices(data=None, prices=None):
    data["timestamp"] = data["time"].apply(lambda x: int(x.replace(second=0).timestamp()))
    assets = list(set(data["asset"].tolist()))
    dfs = []
    for asset in assets:
        joined = data[data["asset"] == asset].reset_index().set_index("timestamp") \
            .join(
                prices[prices["asset"] == asset].drop(columns=["asset"]).reset_index().set_index("timestamp"),
                on="timestamp",
                how="left"
            )
        joined.reset_index(inplace=True)
        dfs.append(
            joined.rename(columns={"close": "price_EUR"}) \
            .drop(columns=["open", "high", "low", "volume", "trades", "timestamp", "index"])
        )
    df = pd.concat(dfs)
    df.loc[df["asset"] == "ZEUR", "price_EUR"] = 1
    df.set_index("refid", inplace=True)
    return df

if __name__ == '__main__':

    data = load_file("ledgers.csv")
    data = prepare(data)

    transactions, changes, staks, deposits = split_types(data)

    df = changes

    assets_files = {
        "USDT": "Kraken_OHLCVT/USDTEUR_1.csv",
        "GRT": "Kraken_OHLCVT/GRTEUR_1.csv",
        "ADA": "Kraken_OHLCVT/ADAEUR_1.csv",
        "OGN": "Kraken_OHLCVT/OGNEUR_1.csv",
        "OXT": "Kraken_OHLCVT/OXTEUR_1.csv",
        "REN": "Kraken_OHLCVT/RENEUR_1.csv",
        "XXBT": "Kraken_OHLCVT/XBTEUR_1.csv",
        "ZEUR": None,
        "XETH": "Kraken_OHLCVT/ETHEUR_1.csv",
        "BNC": "Kraken_OHLCVT/BNCEUR_1.csv",
        "XLTC": "Kraken_OHLCVT/LTCEUR_1.csv"
    }

    print(df)
    assets_prices = build_assets_prices(assets_dict=assets_files)
    df = attach_prices(data=df, prices=assets_prices)

    print(df)
    df = agg_transactions(df)
    
    #df = df[(df["asset_buy"] == "BNC") | (df["asset_sell"] == "BNC")]
    print(df)
    
    assets = list_assets(df)
    dfs = []
    for asset in assets:
        print("Processing asset {}".format(asset))
        asset_founding = balances(data=df, pair=asset)
        asset_founding = attach_buy_prices(data=df, asset_founding=asset_founding)
        dfs.append(asset_founding)
    assets_foundings = pd.concat(dfs)
    df = df.join(assets_foundings, on="refid", how="left")   
        
    # #print(df.to_string())
    print(df)


    
