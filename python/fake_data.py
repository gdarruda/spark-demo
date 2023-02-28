import uuid
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count

import shutil

NUM_CLASSES = 30
BATCH_SIZE = 2_000_000
NUM_CLIENTS = 60_000_000
PATH = '../resources/inline.parquet'

class RandomData:

    def __init__(self,
                 num_rows: int,
                 num_classes: int,
                 path: str) -> None:

        self.num_rows = num_rows
        self.num_classes = num_classes
        self.path = path
    
    def generate_file(self,
                      filename: str):

       df = pd.DataFrame({
            **{"id_client": [str(uuid.uuid4()) for _ in range(self.num_rows)]},
            **{f"class_{c}": np.random.uniform(size=self.num_rows)
            for c in range(self.num_classes)}
        })
        
       df.to_parquet(f"{self.path}/{filename}")

if os.path.isdir(PATH):
    shutil.rmtree(PATH)

os.mkdir(PATH)

rd = RandomData(BATCH_SIZE, NUM_CLASSES, PATH)

with Pool(cpu_count()) as p:
    p.map(rd.generate_file, 
          [str(uuid.uuid4()) 
           for _ in range(NUM_CLIENTS // BATCH_SIZE)])
