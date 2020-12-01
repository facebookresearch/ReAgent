import unittest

from bento.testutil import run_notebook


class NotebookTests(unittest.TestCase):
    def test_reinforce(self):
        path = "reagent/notebooks/REINFORCE_for_CartPole_Control.ipynb"
        variables = run_notebook(path)
        self.assertGreater(variables["mean_reward"], 180)
