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
        self.changes = self.transactions[
            (self.transactions["type"] != "transfer") &
            (self.transactions["type"] != "staking") &
            (self.transactions["type"] != "deposit")
        ]
        # self.changes = self.transactions[
        #     (self.transactions["type"] != "staking") &
        #     (self.transactions["type"] != "transfer")
        # ]
        self.staks = self.transactions[
            (self.transactions["type"] == "transfer") |
            (self.transactions["type"] == "staking")
        ]
        self.deposits = self.transactions[self.transactions["type"] == "deposit"]
        self.assets = list(set(self.changes["asset"].tolist()))
        self.innacurate_prices = None
        self.declarable_assets = None

    @classmethod
    def from_file(KrakenDF, filename=None):
        file_df = load_file(filename=filename)
        file_df["operation"] = file_df["amount"].apply(
            lambda x: "sell" if x < 0 else "buy"
        )
        file_df["time"] = pd.to_datetime(file_df["time"], utc=False)
        for col in ["txid", "refid", "type", "subtype", "asset", "operation"]:
            file_df[col] = file_df[col].astype("string")
        file_df.drop(columns=["subtype", "aclass", "balance"], inplace=True)
        file_df.set_index("refid", inplace=True)
        return KrakenDF(df=file_df)

    def _join_with_nearest_prices(self, prices=None, asset=None):
        # If asset is ZEUR simulate df with pricing history
        if asset == "ZEUR":
            prices = pd.DataFrame(
                [[1000, 1, 1, 1, 1, 1, 1]],
                columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
            )
            prices["asset"] = "ZEUR"
            prices["asset"] = prices["asset"].astype("string")
        df_left = self.changes[self.changes["asset"] == asset] \
            .reset_index() \
            .set_index("timestamp")
        df_right = prices[prices["asset"] == asset] \
            .drop(columns=["asset"]) \
            .reset_index()
        df_right["timestamp_nearest"] = df_right["timestamp"]
        df_right.set_index("timestamp", inplace=True)
        joined = pd.merge_asof(df_left, df_right, on="timestamp", direction="nearest")
        joined.set_index("refid", inplace=True)
        return joined

    def _attach_prices_nearest(self, prices=None):
        """For each asset different from ZEUR, attach the nearest price found, and
        creates a column price_time_diff with the difference in seconds.
        It normal to have less than 60 seconds due to minute agg for prices.
        """
        dfs = []
        # Just attach prices others than ZEUR
        for asset in self.assets:
            joined = self._join_with_nearest_prices(prices=prices, asset=asset)
            joined["time_joined"] = pd.to_datetime(
                joined["timestamp_nearest"].apply(lambda x: pd.Timestamp.fromtimestamp(x)),
                utc=False
            )
            joined["price_time_diff"] = joined["time"] - joined["time_joined"]
            joined["price_time_diff"] = joined["price_time_diff"].apply(lambda x: x.total_seconds())
            joined.drop(columns=["timestamp_nearest"], inplace=True)
            joined.reset_index(inplace=True)
            dfs.append(
                joined.rename(columns={"close": "price_EUR"}) \
                .drop(
                    columns=[
                        "open",
                        "high",
                        "low",
                        "volume",
                        "trades",
                        "index"
                    ]
                )
            )
        df = pd.concat(dfs)
        df.set_index("refid", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        return df

    def attach_prices(self, prices=None):
        self.changes["timestamp"] = self.changes["time"] \
            .apply(lambda x: int(
                x.tz_localize(tz='Europe/Madrid') \
                .floor(freq="T").timestamp())
                   )
        self.changes = self._attach_prices_nearest(prices=prices)
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
        sells = self.changes[self.changes["asset_sell"] == pair] \
                    .sort_values(by="time", ascending=True)
        buys = self.changes[self.changes["asset_buy"] == pair]
        df_refs = []
        for idx, item in sells.iterrows():
            amount = item["amount_sell"]
            origin = buys[buys["time"] < item["time"]] \
                .sort_values(by="time", ascending=True)
            computed_sells = fill_with(abs(amount), origin)

            if computed_sells == []:
                if pair != "ZEUR":
                    logging.error(
                        "ERROR, sells not linked to foundings for asset {}" \
                        .format(pair)
                    )
                    print(buys)
                    print(sells)
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
        df = self.changes.reset_index()
        # Buy costs comes from fifo foundings
        df["buy_cost"] = df["quantity"] * df["price_bought_EUR"]
        df = df.groupby(
            [
                "refid",
                "time",
                "asset_buy",
                "amount_buy",
                "fee_buy",
                "price_EUR_buy",
                "asset_sell",
                "amount_sell",
                "fee_sell",
                "price_EUR_sell"
            ]
        ).sum().reset_index()
        # Sell costs comes from the asset (usually EUR/USDT) bought for each currency
        df["sell_cost"] = df["amount_buy"] * df["price_EUR_buy"]
        self.changes = self.changes.join(
            df.set_index("refid")[["sell_cost", "buy_cost"]],
            on="refid",
            how="left"
        )
        return self

    def build_declarables(self):
        self.declarable_assets = self.changes[self.changes["asset_sell"] != "ZEUR"]
        self.declarable_assets["gain"] = self.declarable_assets["sell_cost"] \
            - self.declarable_assets["buy_cost"]
        self.declarable_assets.drop(
            columns=[
                "refid_foundings",
                "quantity",
                "price_bought_EUR",
                "sell_cost",
                "buy_cost"
            ],
            inplace=True
        )
        self.declarable_assets.drop_duplicates(inplace=True)
        self.declarable_assets.sort_values(by="time", ascending=True)
        return self


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

    print(krakendf.changes)
    krakendf.calculate_costs() \
            .build_declarables()

    df = krakendf.declarable_assets
    print(df)

    # #print(df.to_string())
    #print(krakendf.changes)
