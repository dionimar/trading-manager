import json
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import logging


def load_file(filename=None, **kwargs):
    return pd.read_csv(filename, **kwargs)


class AssetZEUR:
    @staticmethod
    def prices():
        df = pd.DataFrame(
            [[1000, 1, 1, 1, 1, 1, 1]],
            columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
        )
        df["asset"] = "ZEUR"
        df["asset"] = df["asset"].astype("string")
        return df

class InventoryFIFO:
    """Input dataframes must be indexed by 'refid' and contain columns 
    'amount_sell', 'amount_buy' and 'time'.
    """
    @staticmethod
    def inventory_asset(buys=None, sells=None, asset=None):
        df_refs = []
        for idx, item in sells.iterrows():
            amount = item["amount_sell"]
            origin = buys[buys["time"] < item["time"]] \
                .sort_values(by="time", ascending=True)
            computed_sells = InventoryFIFO.fill_with(abs(amount), origin)

            if computed_sells == []:
                logging.error(
                    "ERROR, sells not linked to foundings for asset {}" \
                    .format(asset)
                )
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

    @staticmethod
    def fill_with(amountt=None, options=None):
        assignment = []
        quantity_to_fill = amountt
        for idx, item in options.iterrows():
            if quantity_to_fill > 0:
                if item["amount_buy"] > quantity_to_fill:
                    assignment.append({"idx": idx, "quantity": quantity_to_fill})
                    quantity_to_fill = 0
                else:
                    assignment.append({"idx": idx,  "quantity": item["amount_buy"]})
                    quantity_to_fill -= item["amount_buy"]
        return list(filter(lambda x: abs(x["quantity"]) > 0, assignment))



class KrakenDF:
    def __init__(self, df=None):
        self.df = df
        self.transactions = self._build_fact()
        self.assets = self._get_assets(lst=df["asset"].tolist())
        self.prices = None
        self.declarable_assets = None
        self.inventory = None
        logging.info(f"Found assets {self.assets}")

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

    def _get_assets(self, lst=None):
        return list(set(filter(lambda x: not x.endswith(".S"), lst)))

    def _build_fact(self):
        """Self joins to link assets sells with buys.
        """
        df = self.df[
            self.df["txid"].notnull()
        ][["time", "asset", "amount", "operation"]].copy()
        transactions = df[df["operation"] == "buy"].join(
            df[df["operation"] == "sell"],
            on="refid",
            how="left",
            lsuffix="_buy",
            rsuffix="_sell"
         ).drop(
             columns=["operation_buy", "operation_sell", "time_sell"]
         ).rename(
             columns={"time_buy": "time"}
         )
        # TODO assert time_buy == time_sell except for foundings, stakings
        logging.info(
            "Built transactions data, {} records".format(transactions.shape[0])
        )
        return transactions

    def build_prices(self, prices=None):
        df = self.df[["time", "asset"]].copy()
        df["time_key"] = df["time"]
        df.reset_index(inplace=True)
        df.set_index(["refid", "time_key"], inplace=True)
        for idx, item in df.iterrows():
            df.loc[idx, "timestamp"] = \
                int(parse(str(item["time"])).replace(second=0).timestamp())
        df["timestamp"] = df["timestamp"].astype("int")
        df.reset_index(inplace=True)
        df.drop(columns=["time_key"], inplace=True)
        self._test_timestamp_conversion(df=df)
        dfs = []
        for asset in self.assets:
            df_left = df[df["asset"] == asset].set_index("timestamp")
            df_right = prices[prices["asset"] == asset] \
                .drop(columns=["asset", "open", "high", "low", "volume", "trades"]) \
                .reset_index(drop=True)
            df_right["timestamp_nearest"] = df_right["timestamp"]
            df_right.set_index("timestamp", inplace=True)
            joined = pd.merge_asof(df_left, df_right, on="timestamp", direction="nearest")
            joined.set_index("refid", inplace=True)
            print(joined)
            joined["time_joined"] = pd.to_datetime(
                joined["timestamp_nearest"].apply(lambda x: pd.Timestamp.fromtimestamp(x)),
                utc=False
            )
            joined["price_time_diff"] = joined["time"] - joined["time_joined"]
            joined["price_time_diff"] = joined["price_time_diff"].apply(lambda x: x.total_seconds())
            joined.drop(columns=["timestamp_nearest", "timestamp"], inplace=True)
            dfs.append(joined)
            
        _prices = pd.concat(dfs)
        #self._test_timestamp_conversion()
        # Clean time_joined and price_time_diff for asset EUR
        for idx, item in _prices[_prices["asset"] == "ZEUR"].iterrows():
            _prices.loc[idx, "time_joined"] = _prices.loc[idx, "time"]
            _prices.loc[idx, "price_time_diff"] = 0
        self.prices = _prices.copy()
        print(self.prices.to_string())
        return self

    def _test_timestamp_conversion(self, df=None):
        for idx, item in df.iterrows():
            orig = parse(str(item["time"])).replace(second=0)
            target = datetime.fromtimestamp(item["timestamp"])
            if orig != target:
                print(orig, item["time"], target, item["timestamp"])
                print(item["time"], parse(str(item["time"])).replace(second=0), target)
                raise Exception("Timestamp conversion failed, test not passed")

    def _attach_buy_prices(self, asset_founding=None):
        """Data comes with refid index"""
        if asset_founding is None:
            return None
        asset_indexed = asset_founding.reset_index() \
            .rename(columns={"refid": "refid_origin", "idx": "refid"})
        asset_indexed = asset_indexed.set_index("refid")
        _data = self.transactions[["price_EUR_buy"]]
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

    def build_inventory(self, strategy=InventoryFIFO):
        """Calculates sells distribution for available buys with FIFO method"""
        if self.transactions is None:
            raise Exception("Transactions must be build before building inventory")
        dfs = []
        for asset in self.assets:
            logging.info("Processing asset {}".format(asset))
            sells = self.transactions[self.transactions["asset_sell"] == asset] \
                        .sort_values(by="time", ascending=True)
            buys = self.transactions[self.transactions["asset_buy"] == asset]
            asset_founding = strategy.inventory_asset(
                buys=buys, sells=sells, asset=asset
            )
            dfs.append(asset_founding)
        assets_foundings = pd.concat(dfs).set_index("refid")
        self.inventory = self.transactions.join(assets_foundings, on="refid", how="left")
        return self

    def _test_fifo(self):
        """Checks if quantity used for referencing sells is at most
        the amount_buy in total
        """
        if self.inventory is None:
            raise Exception("Inventory must be computed before tested")
        df_ref = self.inventory.reset_index()[["refid_foundings", "quantity"]]
        df_founds = self.transactions.reset_index()[["refid", "amount_buy"]] \
            .drop_duplicates()
        df_ref = df_ref.groupby("refid_foundings") \
                       .sum() \
                       .reset_index() \
                       .rename(columns={"refid_foundings": "refid"}) \
                       .set_index("refid")
        df_checks = df_ref.join(
            df_founds.set_index("refid"),
            on="refid",
            how="left"
        )
        df_checks["assertion"] = df_checks["quantity"] <= df_checks["amount_buy"]
        check = all(df_checks["assertion"].to_list())
        if not check:
            logging.error(
                "Foundings distribution test failing. The quantity of founds used error"
            )
        logging.info("Test for FIFO distribution passed.")

    def calculate_costs(self):
        if self.inventory is None:
            raise Exception("Inventory strategy must be computed before calculating costs")
        df = self.inventory.reset_index()
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
        self.inventory = self.inventory.join(
            df.set_index("refid")[["sell_cost", "buy_cost"]],
            on="refid",
            how="left"
        )
        self.inventory["fee_buy_EUR"] = \
            (self.inventory["fee_buy"] * self.inventory["price_EUR_buy"]) \
            + (self.inventory["fee_sell"] * self.inventory["price_EUR_sell"])
        return self

    def build_declarables(self):
        if self.inventory is None:
            raise Exception("Inventory must be computed before declarables are computed")
        # Transactions from ZEUR to crypto are not declarable (it's just entering to crypto world)
        # Only declare changes in assets
        self.declarable_assets = self.inventory[self.inventory["asset_sell"] != "ZEUR"].copy()
        self.declarable_assets["gain"] = self.declarable_assets["sell_cost"] \
            - self.declarable_assets["buy_cost"]
        self.declarable_assets.sort_values(by="time", ascending=True)
        logging.info(
            "Built {} declarable assets".format(
                self.declarable_assets.reset_index()[["refid"]].drop_duplicates().shape[0]
            )
        )
        return self

    def agg_declarables(self):
        if self.declarable_assets is None:
            raise Exception("Declarable assets must be computed before agregating them")
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
        self.declarable_assets = self.declarable_assets.reset_index()
        self.declarable_assets = self.declarable_assets[
            ~self.declarable_assets["refid"].str.startswith("Q")
        ].set_index("refid")
        return self


def build_assets_prices(assets_dict=None):
    dfs = []
    for key, value in assets_dict.items():
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
        "USDT": "../Kraken_OHLCVT/USDTEUR_1.csv",
        "GRT": "../Kraken_OHLCVT/GRTEUR_1.csv",
        "ADA": "../Kraken_OHLCVT/ADAEUR_1.csv",
        "OGN": "../Kraken_OHLCVT/OGNEUR_1.csv",
        "OXT": "../Kraken_OHLCVT/OXTEUR_1.csv",
        "REN": "../Kraken_OHLCVT/RENEUR_1.csv",
        "XXBT": "../Kraken_OHLCVT/XBTEUR_1.csv",
        "XETH": "../Kraken_OHLCVT/ETHEUR_1.csv",
        "BNC": "../Kraken_OHLCVT/BNCEUR_1.csv",
        "XLTC": "../Kraken_OHLCVT/LTCEUR_1.csv"
    }


    _prices = build_assets_prices(assets_dict=assets_files)
    assets_prices = pd.concat([_prices, AssetZEUR.prices()])
    

    krakendf = KrakenDF.from_file("ledgers.csv")
    # krakendf.attach_prices(prices=assets_prices)
    # # print("################### prices")
    # # print(krakendf.changes.sort_values(by="refid").to_string())

    krakendf.build_inventory()
    krakendf.build_prices(prices=assets_prices)
    # # print("################### inventory")
    # # print(krakendf.transactions.sort_values(by="refid").to_string())
    # krakendf._test_fifo()
    # krakendf.calculate_costs()
    # # print("################### prices")
    # # print(krakendf.changes.sort_values(by="refid").to_string())
    # krakendf.build_declarables() \
    #         .agg_declarables()

    # df = krakendf.declarable_assets.sort_values(by=["asset_buy", "time"])
    # print(df.to_string())

    # #print(df.to_string())
    # print(krakendf.changes)
