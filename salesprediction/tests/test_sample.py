import unittest
import sample_package.sample_submodule.sample as smp

class Test_SampleModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_helloWorld(self):
        self.assertTrue(smp.helloWorld())

    def tearDown(self):
        return super().tearDown()