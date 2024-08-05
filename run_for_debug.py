
import subprocess

config='projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'
checkpoint = 'ckpt/sparse4dv3_r50.pth'

cmd = f'python ./tools/test.py {config} {checkpoint} --eval bbox'

result = subprocess.run(cmd.split(' '))
