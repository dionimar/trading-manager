import json
from datetime import datetime
from dateutil.parser import parse
import pandas as pd


def load_file(filename=None, **kwargs):
    return pd.read_csv(filename, **kwargs)

def prepare(df=None):
    df["operation"] = df["amount"].apply(lambda x: "sell" if x < 0 else "buy")
    df["time"] = pd.to_datetime(df["time"])
    for col in ["txid", "refid", "type", "subtype", "asset", "operation"]:
        df[col] = df[col].astype("string")
    df.drop(columns=["subtype", "aclass", "balance"], inplace=True)
    #df.set_index("refid", inplace=True)
    return df

def split_types(df=None):
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

def join_operations(df=None):
    sells = df[df["operation"] == "sell"]
    buys = df[df["operation"] == "buy"]
    joined = buys.set_index("refid").join(
        sells.set_index("refid"),
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
    sells = data[data.asset_sell == pair]
    buys = data[data.asset_buy == pair]
    for idx, item in sells.iterrows():
        amount = item["amount_sell"]
        origin = buys[buys.time < item["time"]].sort_values(by="time", ascending=True)
        computed_sells = fill_with(abs(amount), origin)
        for item_dist in computed_sells:
            index, diff = item_dist["idx"], item_dist["quantity"]
            buys.at[index, "amount_buy"] -= diff
        data.at[idx, "founds_come_from"] = computed_sells


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

def attach_price(data=None, prices=None):
    data["timestamp"] = data["time"].apply(lambda x: int(x.replace(second=0).timestamp()))
    assets = list(set(data["asset"].tolist()))
    dfs = []
    for asset in assets:
        joined = data[data["asset"] == asset].set_index("timestamp") \
            .join(
                prices[prices["asset"] == asset].drop(columns=["asset"]).set_index("timestamp"),
                on="timestamp",
                how="left"
            )
        joined.reset_index(inplace=True)
        dfs.append(
            joined.rename(columns={"close": "price_EUR"}) \
            .drop(columns=["open", "high", "low", "volume", "trades", "timestamp"])
        )
    return pd.concat(dfs)

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

    assets_prices = build_assets_prices(assets_dict=assets_files)

    df = attach_price(data=df, prices=assets_prices)
    print(df)
    df = join_operations(df)  
    
    
    df["founds_come_from"] = None
    assets = list_assets(df)
    for asset in assets:
        print("Processing asset {}".format(asset))
        balances(data=df, pair=asset) 

    # #print(df.to_string())
    print(df)   

    
