import csv
import os
import tempfile
import unittest

from utils.log_utils import CsvLogger


class CsvLoggerTest(unittest.TestCase):
    def test_later_metrics_expand_schema_without_losing_earlier_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'metrics.csv')
            logger = CsvLogger(path)
            logger.log({'training/base': 1.0}, step=0)
            logger.log(
                {
                    'training/base': 2.0,
                    'training/actor/transport': 3.0,
                    'training/teacher/q': 4.0,
                },
                step=5,
            )
            logger.close()

            with open(path, newline='') as metric_file:
                rows = list(csv.DictReader(metric_file))

        self.assertEqual(
            list(rows[0]),
            [
                'training/base',
                'step',
                'training/actor/transport',
                'training/teacher/q',
            ],
        )
        self.assertEqual(rows[0]['training/base'], '1.0')
        self.assertEqual(rows[0]['training/actor/transport'], '')
        self.assertEqual(rows[1]['training/actor/transport'], '3.0')
        self.assertEqual(rows[1]['training/teacher/q'], '4.0')
        self.assertEqual(rows[1]['step'], '5')


if __name__ == '__main__':
    unittest.main()
