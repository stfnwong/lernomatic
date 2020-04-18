"""
TEST_SPLITTER
Unit tests for data splitter

Stefan Wong 2019
"""

import argparse
# units under test
from lernomatic.data import data_split


class TestListSplitter:
    verbose = True
    test_split_ratios = [0.7, 0.2, 0.1]
    test_split_names = ['ut_test', 'ut_train', 'ut_val']
    test_split_method = 'random'

    def test_gen_split(self) -> None:
        splitter = data_split.ListSplitter(
            split_ratios    = self.test_split_ratios,
            split_names     = self.test_split_names,
            split_method    = self.test_split_method,
            split_data_root = self.test_data_root,
            verbose         = self.verbose
        )
        print('Generated new splitter object')
        print(splitter)

        print('Generating data list....', end=' ')
        cat_list = os.listdir(GLOBAL_OPTS['test_data_root'] + '/Cat')
        dog_list = os.listdir(GLOBAL_OPTS['test_data_root'] + '/Dog')
        print('done')

        print('Generating label list....', end=' ')
        label_list = [0] * len(cat_list) +\
                     [1]* len(dog_list)
        print('done')

        print('Splitting...')
        splits = splitter.gen_splits(
            cat_list + dog_list,
            label_list
        )

        print('done\n')
        for n, s in enumerate(splits):
            print('Split %d <%s> : length : %d' % (n+1, s.get_name(), len(s)))

        # TODO : need to write some asserts
