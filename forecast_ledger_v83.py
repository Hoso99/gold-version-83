from forecast_ledger_v825 import ForecastLedger


class ForecastLedgerV83(ForecastLedger):
    def __init__(self, path="forecast_ledger_v83.sqlite"):
        super().__init__(path)

