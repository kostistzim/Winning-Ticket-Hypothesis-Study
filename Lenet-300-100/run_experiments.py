import subprocess

TRIALS = range(5)
MODES = ["iterative", "oneshot"]
REINITS = [False, True]

for mode in MODES:
    for random_reinit in REINITS:
        for trial in TRIALS:
            tag = f"{mode}_{'random' if random_reinit else 'dense'}_trial{trial}"
            print(f"🚀 Launching {tag}")

            cmd = [
                "uv", "run", "main.py",
                "--trial", str(trial),
                "--mode", mode,
            ]

            if random_reinit:
                cmd.append("--random_reinit")

            subprocess.run(cmd, check=True)
