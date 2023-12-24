import torch
import unittest
import tempfile
from lstm import create_model, load, LSTMWrapper

class TestLSTMModel(unittest.TestCase):

    def test_model_initialization(self):
        input_size = 10
        hidden_layer_size = 20
        output_size = 5
        model = create_model(input_size, hidden_layer_size, output_size).model
        self.assertEqual(model.lstm.input_size, input_size)
        self.assertEqual(model.lstm.hidden_size, hidden_layer_size)
        self.assertEqual(model.linear.out_features, output_size)

    def test_forward_pass(self):
        model = create_model(10, 20, 5).model
        input_seq = torch.randn(5, 10)  # 5 time steps, 10 features
        output = model(input_seq)
        self.assertEqual(output.shape, (5,5,))

    def test_training(self):
        wrapper = create_model(10, 20, 1)
        input_seq = torch.randn(10, 10)  # 10 time steps, 10 features
        label_seq = torch.randn(10, 1)   # Corresponding 10 labels
        losses = wrapper.train(input_seq, label_seq, 100)
        self.assertLess(losses[-1], losses[0])  # Loss should decrease

    def test_prediction(self):
        wrapper = create_model(10, 20, 1)
        input_seq = torch.randn(10, 10)
        prediction = wrapper.predict(input_seq)
        self.assertIsInstance(prediction, torch.Tensor)
        self.assertEqual(prediction.shape, (10,1))

    def test_save_load(self):
        model = create_model(10, 20, 5)
        model.save('test_model.pth')
        loaded_model = load('test_model.pth')
        self.assertIsInstance(loaded_model, LSTMWrapper)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper = create_model(10, 20, 5)
            temp_model_path = f'{tmp_dir}/test_model.pth'
            wrapper.save(temp_model_path)
            loaded_model = load(temp_model_path)
            self.assertIsInstance(loaded_model, LSTMWrapper)

if __name__ == '__main__':
    unittest.main()
