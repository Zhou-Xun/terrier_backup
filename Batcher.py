import pyterrier as pt

class Batcher(pt.Transformer):

    def __init__(self, parent, batch_size=50):
        self.parent = parent
        self.batch_size = batch_size

    def transform(self, inputRes):
        import math, pandas as pd
        from pyterrier.model import split_df
        from tqdm import tqdm
        num_chunks = math.ceil(len(inputRes) / self.batch_size)
        iterator = split_df(inputRes, num_chunks)
        iterator = tqdm(iterator, desc="pt.apply", unit='row')
        rtr = pd.concat([self.parent.transform(chunk_df) for chunk_df in iterator])
        return rtr