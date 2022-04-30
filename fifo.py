import json
from datetime import datetime
from dateutil.parser import parse
import pandas as pd


def load_file(filename=None):
    return pd.read_csv(filename)

def prepare(df=None):
    df["operation"] = df["amount"].apply(lambda x: "sell" if x < 0 else "buy")
    df["time"] = pd.to_datetime(df["time"])
    df["timestamp"] = df["time"].apply(lambda x: int(x.replace(second=0).timestamp()))
    for col in ["txid", "refid", "type", "subtype", "asset", "operation"]:
        df[col] = df[col].astype("string")
    df.drop(columns=["subtype", "aclass", "balance"], inplace=True)
    df.set_index("refid", inplace=True)
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
    print(buys.dtypes)
    print(sells.dtypes)
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
            "timestamp_sell",
            "operation_buy",
            "operation_sell"
        ]
    ).rename(columns={"time_buy": "time", "timestamp_buy": "timestamp"})






    
def fill_with(amountt=None, options=None):
    # Sort by time, asc
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
    data["founds_come_from"] = None
    sells = data[data.asset_sell == pair]
    #sells.set_index("refid", inplace=True)
    buys = data[data.asset_buy == pair]
    #buys.set_index("refid", inplace=True)
    print(buys)
    print(sells)
    for idx, item in sells.iterrows():
        amount = item["amount_sell"]
        origin = buys[buys.time < item["time"]]
        computed_sells = fill_with(abs(amount), origin)
        for item_dist in computed_sells:
            index, diff = item_dist["idx"], item_dist["quantity"]
            buys.at[index, "amount_buy"] -= diff
        print(computed_sells)
        data.at[idx, "founds_come_from"] = computed_sells


def list_assets(data=None):
    assets_l = data["asset_buy"]
    assets_r = data["asset_sell"]
    assets = assets_l.values.tolist() + assets_r.values.tolist()
    return list(set(assets))

    


if __name__ == '__main__':

    #sells = load_file(filename="computed_sells/part-00000-a66c1803-d568-4b0e-a3ca-b64af5f44279-c000.json")
    #sells = list(filter(lambda x: x["asset_left"] == "BNC" or x["asset_right"] == "BNC", sells))
    data = load_file("ledgers.csv")
    data = prepare(data)
    print(data)

    transactions, changes, staks, deposits = split_types(data)

    df = join_operations(changes)
    print(df)

    
    # data = pd.read_json("computed_sells/part-00000-a66c1803-d568-4b0e-a3ca-b64af5f44279-c000.json", orient="records", lines=True)
    # data.drop(labels=["balance_left", "balance_right", "txid_left", "txid_right", "pair"], axis=1, inplace=True)
    # data.set_index("refid", inplace=True)
    
    
    
    df = df[(df.asset_buy == "BNC") | (df.asset_sell == "BNC")]
    # #print(data)

    assets = list_assets(df)
    
    #for asset in assets:
    balances(data=df, pair="BNC")

    print(df)

    
