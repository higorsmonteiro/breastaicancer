import random
from tqdm import tqdm
from pathlib import Path
from pydantic import ValidationError

from hapcancer.schemas.validation_models import *
from hapcancer.config_manager import ConfigInterface

class DataValidator(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)

        self.filepaths = {
            "anamnesis": list(self.extract_path.joinpath(self.extract_folders['anamnesis']).glob("*.parquet")),
            "mammogram_exams": list(self.extract_path.joinpath(self.extract_folders['mammogram_exams']).glob("*.parquet")),
            "user": list(self.extract_path.joinpath(self.extract_folders['user_person_data']).glob("user*.parquet")),
            "patient": list(self.extract_path.joinpath(self.extract_folders['user_person_data']).glob("patient*.parquet")),
            "person": list(self.extract_path.joinpath(self.extract_folders['user_person_data']).glob("person*.parquet")),
            "biopsy": list(self.extract_path.joinpath(self.extract_folders['biopsy']).glob("*.parquet")),
        }
        self.valid_rows = {
            "anamnesis": 0, "mammogram_exams": 0, "user": 0, "patient": 0, "person": 0, "biopsy": 0
        }
        self.total_rows = {
            "anamnesis": 0, "mammogram_exams": 0, "user": 0, "patient": 0, "person": 0, "biopsy": 0
        }
        self.errors = {
            "anamnesis": [], "mammogram_exams": [], "user": [],  "patient": [], "person": [], "biopsy": []
        }
        self.datamodels = {
            "anamnesis": RawAnamnesisData,
            "mammogram_exams": RawMammogramData,
            "user": RawUserData,
            "patient": RawPatientData,
            "person": RawPersonData,
            "biopsy": RawBiopsyData
        }

    def validate_raw_data(self, fraction : Optional[float] = 0.5):
        for cur_data_class, cur_filepaths in self.filepaths.items():
            print(f"Validation for {cur_data_class} ... ", end='')
            # -- select the proper data model for validation
            cur_datamodel = self.datamodels[cur_data_class]
            
            # -- validate only a fraction of files according to 'fraction'.
            k_to_sample = int(len(cur_filepaths)*fraction)
            if k_to_sample<1: k_to_sample = 1
            sampled_filepaths = random.sample(cur_filepaths, k=k_to_sample)
            
            for current_filepath in sampled_filepaths:
                cur_df = pd.read_parquet(current_filepath) # -- logic should change when type of persistance also changes
                
                # -- validation
                for ix, row in cur_df.iterrows():
                    self.total_rows[cur_data_class]+=1
                    try:
                        valid_row = cur_datamodel(**row.to_dict())
                        self.valid_rows[cur_data_class]+=1
                    except ValidationError as e:
                        self.errors[cur_data_class].append({
                            "row": ix, "file": current_filepath.stem,
                            "error": e.errors()
                        })
            print("done.")

        print(f"Total rows: {self.total_rows}")
        print(f"Valid rows: {self.valid_rows}")
        for k, v in self.errors.items():
            print(f"Total errors for {k}: {len(self.errors[k])}")
        for k, v in self.errors.items():
            if len(self.errors[k]):
                k_to_sample = 1 if len(self.errors[k]) <=2 else 3
                print(f"Sample errors for {k}: {random.sample(self.errors[k], k=k_to_sample)}")

    

