class ColectedData:
    def __init__(self,
                 value,
                 description: str,
                 link: str,
                 source: str):
        self.value = value
        self.description = description
        self.link = link
        self.source = source


class DatabaseWitnessCore:
    """Stocke les valeurs utilisées dans witness core"""
    # Example :
    InvestFossil2020 = ColectedData(750,
                                    description="Investment in fossil in 2020 in Trillion US$",
                                    link="lol.com",
                                    source="IEA rapport XXXX")

