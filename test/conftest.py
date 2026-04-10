from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    # cleanup code after tests

    # remove pytest and OpenMDAO report output directories from cwd
    for pattern in ("pytest*_out", "__main__*_out"):
        for out_dir in Path().glob(pattern):
            for root, dirs, files in out_dir.walk(top_down=False):  # walk the directory
                for name in files:
                    (root / name).unlink()  # remove subdirectory files, and
                for name in dirs:
                    (root / name).rmdir()  # remove subdirectories
            out_dir.rmdir()  # then remove that tempdir
