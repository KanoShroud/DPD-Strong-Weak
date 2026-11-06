"""Focused output contract tests, no localization work."""
import json
import unittest

import numpy as np

from output_json import dumps


class OutputTests(unittest.TestCase):
    def test_nested(self):
        value={'中文':[{'b':np.bool_(True),'i':np.int64(9),'f':np.float32(1.25)},
                      (np.uint32(4),np.array([[1,2],[3,4]],dtype=np.int64)),None]}
        restored=json.loads(dumps(value,ensure_ascii=False))
        self.assertEqual(restored,{'中文':[{'b':True,'i':9,'f':1.25},[4,[[1,2],[3,4]]],None]})

    def test_actual_count_expression(self):
        count=sum(np.float64(v)<=6 for v in [1.,2.,3.])
        self.assertIsInstance(count,np.integer)
        self.assertEqual(json.loads(dumps({'count':count})),{'count':3})

    def test_nonfinite_rejected(self):
        for value in [float('nan'),np.float32('inf'),np.array([1.,-np.inf])]:
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    dumps({'nested':[value]})

    def test_complex_not_silently_changed(self):
        for value in [1+2j,np.complex64(1+2j),np.array([1+2j])]:
            with self.subTest(value=str(value)):
                with self.assertRaises(TypeError):
                    dumps({'nested':[value]})

    def test_native_representation_unchanged(self):
        value={'a':1.,'b':False,'c':[4,None]}
        self.assertEqual(dumps(value,indent=2),json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':
    unittest.main()
