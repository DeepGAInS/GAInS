import glob
import os
import subprocess

# 指定你的图片文件夹路径
image_folder = '/work/home/acvwd4uw3y181/rsliu/data000/test'

# 指定你的模型权重文件和配置文件路径
weights = '/work/home/acvwd4uw3y181/rsliu/exp_used/CN_ws8_f0.8_ol0.1/model_0006999.pth'
config = '/work/home/acvwd4uw3y181/rsliu/exp_used/CN_ws8_f0.8_ol0.1/config.yaml'

output = '/work/home/acvwd4uw3y181/rsliu/picture/nuclei/ours'
# 使用glob找到所有的图片
images = glob.glob(os.path.join(image_folder, '*'))

# 对每一张图片运行demo
for image in images:
    filename = os.path.splitext(os.path.basename(image))[0] + '_demo'
    command = f"python demo.py --config-file {config} --input {image}  --output {os.path.join(output, filename)} --opts MODEL.WEIGHTS {weights} MODEL.DEVICE cpu"
    subprocess.check_call(command, shell=True)
