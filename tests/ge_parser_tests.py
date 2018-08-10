import unittest

class test_GE_parser(unittest.TestCase):

    def test_align_basic(self):
        """
        this tests a basic example of the align_sets_by_date
        """
        df1 = pd.DataFrame({"date":[], "price":[1,2,3,2,1]})
