import subprocess
import argparse

from pathlib import Path

parser = argparse.ArgumentParser(description='Optimize edges to get normals')
parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
args = parser.parse_args()
curve_file = args.curve_file



curve_name = Path(curve_file).stem

temp1 = 'temp/temp1.obj'
process1 = subprocess.Popen(['python', 'curve_to_obj.py', str(curve_file), temp1])
process1.wait()  # Wait for completion



temp2 = 'temp/temp2.obj'
process2 = subprocess.Popen(['python', 't2f_center_and_scale.py', temp1, temp2])
process2.wait()  # Wait for completion




temp3 = 'temp/temp3.obj'
process3 = subprocess.Popen(['python', 'curve_duplicate.py', temp2, temp3])
process3.wait()  # Wait for completion



temp4 = 'temp/temp4.obj'
process4 = subprocess.Popen(['python', 'curve_split_preprocess_split_only.py', temp3, temp4])
process4.wait()  # Wait for completion


temp5 = 'temp/temp5.obj'
process6 = subprocess.Popen(['python', 'curve_resample_preprocess.py', temp4, temp5])
process6.wait()  # Wait for completion



# temp6 = 'temp/temp6.obj'
# process6 = subprocess.Popen(['python', 'curve_duplicate.py', temp5, temp6])
# process6.wait()  # Wait for completion

process7 = subprocess.Popen(['python', 'curve_split_preprocess.py', temp5, '3d-sketches/mingzou/' + curve_name + '.obj'])
process7.wait()  # Wait for completion


