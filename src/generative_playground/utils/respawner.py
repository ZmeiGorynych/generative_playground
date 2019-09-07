import sys, subprocess
print("respawner starting with ", sys.argv[1:])
counter = 0
while True:
    print('spawning, iteration', counter)
    retval = subprocess.run(sys.argv[1:])
    if retval.returncode <0:
        break
    counter += 1