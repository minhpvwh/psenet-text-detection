# import package
import psenet_text_detector as psenet

# set image path and export folder directory
image_path = 'figures/092044508.jpg'
output_dir = 'outputs/'

# apply craft text detection and export detected regions to output directory
prediction_result = psenet.detect_text(image_path, output_dir, cuda=False)