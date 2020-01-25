import glob, os
for file in sorted(glob.glob("*.flac")):
    print("#"*40)
    print(file)
    os.system("python audioAnalysis.py silenceRemoval -i " + file + " --smoothing 0.2 --weight 0.1")
