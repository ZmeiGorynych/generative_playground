import sys, subprocess
print("respawner starting with ", sys.argv[1:])
counter = 0
while True:
    print('spawning', counter)
    subprocess.run(sys.argv[1:])
    counter += 1