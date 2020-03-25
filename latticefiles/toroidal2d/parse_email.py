import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file',help='The source email to parse, in a file')
args = parser.parse_args()


body = open(args.file,"r").read()

jobs = re.split(r'JOB:\ [0-9]*.', body)


for job in jobs[1:]:
    name=re.findall(r'Name: .*[0-9]{8}.txt',job)[0]
    name=re.findall(r'[0-9]+x[0-9]+xxx[a-z|A-Z]+xxx[0-9]{8}', name)
    if len(name) < 1:
        continue
    path=re.split(r'x{3}', name[0])

    energy = re.findall(r'Energy \(per spin\): \-?[0-9]*[\.]{1}[0-9]+', job)[0]
    energy = re.findall(r'\-?[0-9]*[\.]{1}[0-9]+', energy)[0]

    file_to_update = os.path.join(*path, "gs_energy")

    print(f"Updating {file_to_update} with energy {energy}")

    with open(file_to_update, 'w') as f:
        f.write(energy)






