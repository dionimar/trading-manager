import json
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import logging


def load_file(filename=None, **kwargs):
    return pd.read_csv(filename, **kwargs)


class KrakenDF:
    def __init__(self, df=None):
        self.df = df
        self.transactions = df[df["txid"].notnull()]
        # self.changes = self.transactions[
        #     (self.transactions["type"] != "transfer") &
        #     (self.transactions["type"] != "staking") &
        #     (self.transactions["type"] != "deposit")
        # ]
        self.changes = self.transactions[
            (self.transactions["type"] != "staking") &
            (self.transactions["type"] != "transfer")
        ]
        self.staks = self.transactions[
            (self.transactions["type"] == "transfer") |
            (self.transactions["type"] == "staking")
        ]
        self.deposits = self.transactions[self.transactions["type"] == "deposit"]
        self.assets = list(set(self.changes["asset"].tolist()))
        
    @classmethod
    def from_file(KrakenDF, filename=None):
        file_df = load_file(filename=filename)
        file_df["operation"] = file_df["amount"].apply(lambda x: "sell" if x < 0 else "buy")
        file_df["time"] = pd.to_datetime(file_df["time"])
        for col in ["txid", "refid", "type", "subtype", "asset", "operation"]:
            file_df[col] = file_df[col].astype("string")
        file_df.drop(columns=["subtype", "aclass", "balance"], inplace=True)
        file_df.set_index("refid", inplace=True)
        return KrakenDF(df=file_df)

    def attach_prices(self, prices=None):
        self.changes["timestamp"] = self.changes["time"].apply(lambda x: int(x.replace(second=0).timestamp()))
        dfs = []
        for asset in self.assets:
            df_left = self.changes[self.changes["asset"] == asset] \
                          .reset_index() \
                          .set_index("timestamp")
            df_right = prices[prices["asset"] == asset] \
                .drop(columns=["asset"]) \
                .reset_index() \
                .set_index("timestamp")
            joined = df_left.join(df_right, on="timestamp", how="left")
            joined.reset_index(inplace=True)
            dfs.append(
                joined.rename(columns={"close": "price_EUR"}) \
                .drop(columns=["open", "high", "low", "volume", "trades", "timestamp", "index"])
            )
        df = pd.concat(dfs)
        df.loc[df["asset"] == "ZEUR", "price_EUR"] = 1
        df.set_index("refid", inplace=True)
        self.changes = df
        return self
        
    def agg_transactions(self):
        """Self joins to link assets sells with buys.
        """
        self.changes = self.changes[self.changes["operation"] == "buy"].join(
            self.changes[self.changes["operation"] == "sell"],
            on="refid",
            how="left",
            lsuffix="_buy",
            rsuffix="_sell"
        ).drop(
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
        return self

    def _attach_buy_prices(self, asset_founding=None):
        """Data comes with refid index"""
        if asset_founding is None:
            return None
        asset_indexed = asset_founding.reset_index() \
            .rename(columns={"refid": "refid_origin", "idx": "refid"})
        asset_indexed = asset_indexed.set_index("refid")
        _data = self.changes[["price_EUR_buy"]]
        joined = asset_indexed.join(_data, on="refid", how="left") \
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

    def _inventory_fifo_asset(self, pair=None):
        """Returns a df with refid as key and columns for each refid which the founds come from
        also with quantity.
        """
        sells = self.changes[self.changes["asset_sell"] == pair]
        buys = self.changes[self.changes["asset_buy"] == pair]
        df_refs = []
        for idx, item in sells.iterrows():
            amount = item["amount_sell"]
            origin = buys[buys["time"] < item["time"]].sort_values(by="time", ascending=True)
            computed_sells = fill_with(abs(amount), origin)
            
            if computed_sells == []:
                logging.error("ERROR, sells not linked to foundings for asset {}".format(pair))
                continue
            # Update buys with already used balances
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

    def inventory_fifo(self):
        dfs = []
        for asset in self.assets:
            logging.info("Processing asset {}".format(asset))
            asset_founding = self._inventory_fifo_asset(pair=asset)
            asset_founding = self._attach_buy_prices(asset_founding=asset_founding)
            dfs.append(asset_founding)
        assets_foundings = pd.concat(dfs)
        self.changes = self.changes.join(assets_foundings, on="refid", how="left")
        return self

    def calculate_costs(self):
        self.changes["buy_cost"] = self.changes["quantity"] * self.changes["price_bought_EUR"]
        self.changes["sell_cost"] = self.changes[""]

    





    
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


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

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

    krakendf = KrakenDF.from_file("ledgers.csv")
    krakendf.attach_prices(prices=assets_prices) \
            .agg_transactions() \
            .inventory_fifo()
        
    # #print(df.to_string())
    print(krakendf.changes)


    
