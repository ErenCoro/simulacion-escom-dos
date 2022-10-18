from abc import ABC, abstractmethod


class Process_Data(ABC):
 
    def list_features(self):
        raise NotImplementedError("This method hasn't been implemented yet")


    def list_labels(self):
        raise NotImplementedError("This method hasn't been implemented yet")

    def new_dataset(self):
        raise NotImplementedError("This method hasn't been implemented yet")
