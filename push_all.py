import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error:\n{res.stderr}")
    else:
        print(res.stdout)

# Add and commit on main
run("git checkout main")
run("git add .")
run('git commit -m "feat: implement Option A dynamic skills API and cleanup"')
run("git push origin main")

# Update vasudevm
run("git checkout vasudevm")
run("git merge main -m 'Merge main into vasudevm'")
run("git push origin vasudevm")

# Update jagadeesh
run("git checkout jagadeesh")
run("git merge main -m 'Merge main into jagadeesh'")
run("git push origin jagadeesh")

# Update bhanus
run("git fetch origin")
run("git checkout -b bhanus origin/bhanus")
run("git merge main -m 'Merge main into bhanus'")
run("git push origin bhanus")

# Go back to main
run("git checkout main")
