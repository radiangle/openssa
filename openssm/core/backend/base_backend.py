from .abstract_backend import AbstractBackend
from ..inferencer.abstract_inferencer import AbstractInferencer


class BaseBackend(AbstractBackend):
    def __init__(self):
        self.facts = set()
        self.inferencers = set()
        self.heuristics = set()

    def process(self, conversation_id: str, user_input: str):
        pass

    def load_all(self):
        """
        The base backend does not load anything.
        It gets all its facts, inferencers, and heuristics
        through the add_* methods.
        """
        pass

    def add_fact(self, fact: str):
        self.facts.add(fact)

    def add_inferencer(self, inferencer: AbstractInferencer):
        self.inferencers.add(inferencer)

    def add_heuristic(self, heuristic: str):
        self.heuristics.add(heuristic)

    def list_facts(self):
        return self.facts

    def list_inferencers(self):
        return self.inferencers

    def list_heuristics(self):
        return self.heuristics

    def select_facts(self, criteria):
        """
        The base backend simply returns all facts.
        """
        return self.list_facts()

    def select_inferencers(self, criteria):
        """
        The base backend simply returns all inferencers.
        """
        return self.list_inferencers()

    def select_heuristics(self, criteria):
        """
        The base backend simply returns all heuristics.
        """
        return self.list_heuristics()
