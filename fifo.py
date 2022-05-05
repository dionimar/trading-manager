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
        KrakenDFTester._test_timestamp_conversion(df=df)
        dfs = []
        
        for asset in self.assets:
            df_left = df[df["asset"] == asset].set_index("timestamp")
            df_right = prices[prices["asset"] == asset] \
                .drop(columns=["asset", "open", "high", "low", "volume", "trades"]) \
                .reset_index(drop=True) \
                .rename(columns={"close": "price"})
            df_right["timestamp_nearest"] = df_right["timestamp"]
            df_right.set_index("timestamp", inplace=True)
            joined = pd.merge_asof(df_left, df_right, on="timestamp", direction="nearest")
            joined.set_index("refid", inplace=True)
            joined["time_nearest"] = pd.to_datetime(
                joined["timestamp_nearest"].apply(lambda x: pd.Timestamp.fromtimestamp(x)),
                utc=False
            )
            if asset == "ZEUR":
                # Clean time_joined and price_time_diff for asset EUR
                joined["time_delta"] = 0
                joined["time_nearest"] = joined["time"]
            else:
                joined["time_delta"] = joined["time"] - joined["time_nearest"]
                joined["time_delta"] = joined["time_delta"].apply(lambda x: x.total_seconds())
            joined.drop(columns=["timestamp_nearest", "timestamp"], inplace=True)
            joined["time_delta"] = joined["time_delta"].astype("int")
            dfs.append(joined)
            
        _prices = pd.concat(dfs)
        self.prices = _prices.copy()
        return self

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
        self.inventory.rename(columns={"idx": "refid_foundings"}, inplace=True)
        return self

    def calculate_costs(self):
        if self.inventory is None:
            raise Exception("Inventory strategy must be computed before calculating costs.")
        if self.prices is None:
            raise Exception("No prices found to calculate costs.")

        df = self.transactions.reset_index()

        prices = self.prices.reset_index().drop(columns=["refid"])

        df_prices = df.set_index(["time", "asset_buy"]).join(
            prices.rename(columns={"asset": "asset_buy"}).set_index(["time", "asset_buy"]),
            on=["time", "asset_buy"],
            how="left"
        ).rename(
            columns={
                "price": "price_buy",
                "time_nearest": "time_nearest_buy",
                "time_delta": "time_delta_buy"
            }
        ).reset_index().set_index(["time", "asset_sell"]).join(
            prices.rename(columns={"asset": "asset_sell"}).set_index(["time", "asset_sell"]),
            on=["time", "asset_sell"],
            how="left"
        ).rename(
            columns={
                "price": "price_sell",
                "time_nearest": "time_nearest_sell",
                "time_delta": "time_delta_sell"
            }
        ).reset_index().set_index("refid")

        df_inventory = self.inventory.reset_index() \
            .set_index("refid_foundings").join(
            df_prices.reset_index() \
                     .rename(
                         columns={"refid": "refid_foundings"}
                     ).set_index("refid_foundings")[["price_buy"]] \
                     .rename(columns={"price_buy": "price_sell"}),
            on="refid_foundings",
            how="left"
        ).reset_index().set_index("refid").join(
            df_prices[["price_buy"]],
            on="refid",
            how="left"
        )
        
        # buy costs comes from selling asset 
        df_inventory["buy_cost"] = df_inventory["amount_buy"] * df_inventory["price_buy"]
        # sell cost comes from foundings (how much we paid for buying them).
        # Instead of calculating from transaction time, we take the price from founding boughts
        df_inventory["sell_cost"] = df_inventory["quantity"] * df_inventory["price_sell"]

        df_inventory = df_inventory \
            .reset_index()[["refid", "buy_cost", "sell_cost"]] \
            .groupby(["refid", "buy_cost"]).sum().reset_index().set_index("refid")
        return df_inventory.copy()

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


class KrakenDFTester:
    @staticmethod
    def _test_fifo(krakenDF=None):
        """Checks if quantity used for referencing sells is at most
        the amount_buy in total
        """
        if krakenDF.inventory is None:
            raise Exception("Inventory must be computed before tested")
        df_ref = krakenDF.inventory.reset_index()[["refid_foundings", "quantity"]]
        df_founds = krakenDF.transactions.reset_index()[["refid", "amount_buy"]] \
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

    @staticmethod
    def _test_timestamp_conversion(df=None):
        for idx, item in df.iterrows():
            orig = parse(str(item["time"])).replace(second=0)
            target = datetime.fromtimestamp(item["timestamp"])
            if orig != target:
                raise Exception("Timestamp conversion failed, test not passed")


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
    
    krakendf.build_inventory()
    krakendf.build_prices(prices=assets_prices)
    df_costs = krakendf.calculate_costs()
    
    print(krakendf.transactions.sort_values(by=["refid"]).to_string())
    print(krakendf.inventory.sort_values(by=["refid"]).to_string())
    print(krakendf.prices.sort_values(by=["refid"]).to_string())

    
    # krakendf.build_declarables() \
    #         .agg_declarables()

    # df = krakendf.declarable_assets.sort_values(by=["asset_buy", "time"])
    # print(df.to_string())

    

    KrakenDFTester._test_fifo(krakendf)
