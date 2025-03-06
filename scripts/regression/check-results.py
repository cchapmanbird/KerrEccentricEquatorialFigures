import argparse
import numpy as np
import logging
from glob import glob
from itertools import combinations

parser = argparse.ArgumentParser(
    description="Regression testing script to ensure that our waveform models are unchanged after code changes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--errortol", help="relative error tolerance", type=float, default=1e-14
)


def main(rel_tol=1e-14):

    results_files = glob(f"result_*.npy")

    results = {}
    for f in results_files:
        split_f = f.split("_")[1:]
        if len(split_f) > 1:
            env_name = "_".join(split_f)[:-4]
        else:
            env_name = split_f[-1][:-4]

        results[env_name] = np.load(f)

    logging.info(f"Regression test check.")

    keypairs = list(combinations(results, 2))
    test_data = [((a, results[a]), (b, results[b])) for a, b in keypairs]

    for (ka, a), (kb, b) in test_data:
        try:
            np.testing.assert_allclose(a, b, rtol=rel_tol, err_msg="failure")
        except AssertionError:
            logging.error(
                f"Error tolerance {rel_tol:e} exceeded in environments {ka} and {kb}"
            )
            continue
        else:
            logging.info(
                f"Environment pair passed with relative tolerance {rel_tol:e}: {ka} and {kb}"
            )


if __name__ == "__main__":
    args = parser.parse_args()

    approximant = args.approximant
    rel_tol = args.errortol

    logging_output_filename = f"waveform-regression-check-results.log"
    logging.basicConfig(
        filename=logging_output_filename,
        encoding="utf-8",
        level=logging.INFO,
        filemode="w",
    )

    main(approximant, rel_tol=rel_tol)
