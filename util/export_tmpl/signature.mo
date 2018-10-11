{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [0, 3, {{MODEL_IMAGE_SIZE}}, {{MODEL_IMAGE_SIZE}}],
      "rgb_mean": [{{RGB_MEAN}}],
      "rgb_std": [{{RGB_STD}}]
    }
  ],
  "input_type": "image/jpeg",
  "outputs": [
    {
      "data_name": "softmax",
      "data_shape": [0, {{NUM_CLASSES}}]
    }
  ],
  "output_type": "application/json"
}
