import json
import os

from dotenv import load_dotenv
from mpcontribs.client import Client
from tqdm import tqdm


class ContribScreeningCallbacks:
    @staticmethod
    def bgap_greater_than(value):
        return lambda d: d["data"]["EgPBE"]["value"] > value

    @staticmethod
    def synthesized():
        return lambda d: d["data"]["synthesized"]

    @staticmethod
    def lcd_greater_than(value):
        return lambda d: d["data"]["lcd"]["value"] > value

    @staticmethod
    def pld_greater_than(value):
        return lambda d: d["data"]["pld"]["value"] > value


class MOFContributions:
    def __init__(
        self, api_key, project="qmof", host="contribs-api.materialsproject.org"
    ):
        self.client = Client(apikey=api_key, host=host, project=project)
        self.callbacks = ContribScreeningCallbacks()
        self.save_directory = "qmof_dataset"

    def query_contributions(self, num_atoms):
        query = {"num_atoms": num_atoms}
        result = self.client.download_contributions(query=query, include=["tables"])
        return result

    def screen_entries(self, data, callbacks):
        return [d for d in data if all(callback(d) for callback in callbacks)]

    def save_data_as_json(self, data, filename):
        os.makedirs(self.save_directory, exist_ok=True)
        filepath = os.path.join(self.save_directory, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_structures_from_data(self, data):
        json_data = []
        for d in tqdm(data):
            _id = d["structures"][0]["id"]
            structure = self.client.get_structure(_id)
            structure_data = structure.as_dict()
            json_data.append(
                {"id": _id, "structure": structure_data, "data": d["data"]}
            )
        self.save_data_as_json(json_data, "screened_mofs.json")


def main():
    load_dotenv()
    apikey = os.getenv("MP_API_KEY")
    mof_contribs = MOFContributions(api_key=apikey)

    num_atoms = 150
    contributions = mof_contribs.query_contributions(num_atoms)

    print(f"Number of queried qmofs {len(contributions)} with {num_atoms} atoms")

    callbacks = [
        mof_contribs.callbacks.synthesized(),
    ]

    screened_contributions = mof_contribs.screen_entries(contributions, callbacks)
    print(f"Number of screened qmofs {len(screened_contributions)}")
    mof_contribs.get_structures_from_data(screened_contributions)


if __name__ == "__main__":
    main()
