from datetime import date

class ColectedData:
    def __init__(self,
                 value,
                 description: str,
                 link: str,
                 source: str,
                 last_update_date: date):
        self.value = value
        self.description = description
        self.link = link
        self.source = source
        self.last_update_date = last_update_date


class DatabaseWitnessCore:
    """Stocke les valeurs utilisées dans witness core"""
    # Example :
    InvestFossil2020 = ColectedData(750,
                                    description="Investment in fossil in 2020 in Trillion US$",
                                    link="lol.com",
                                    source="IEA rapport XXXX",
                                    last_update_date=date(2023, 10, 13))

