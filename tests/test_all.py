import unittest

import torch

from l4cv.models.vgg16_classifier import VGG16Classifier 

class TestVGG16(unittest.TestCase):

    def setUp(self):
        pass

    def test_vg16(self):

        model = VGG16Classifier()

        x = torch.rand(2,3,32,32)

        y = model(x)

        self.assertEqual(y.shape[-1], 10)


if __name__ == "__main__":

    unittest.main()
