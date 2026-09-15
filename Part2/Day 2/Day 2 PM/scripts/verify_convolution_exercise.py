"""Verify the three Day 2 convolution exercise questions.

The worksheet uses 5x5 inputs and 3x4 kernels.  With valid/no-padding
operation, each result is 3x2.  By default the script reports the
cross-correlation convention used by most CNN libraries.  It also reports
mathematical convolution, where the kernel is flipped before multiplication.
"""

from __future__ import annotations

import numpy as np


QUESTIONS = [
    {
        "input": [
            [1, 0, 1, 0, 1],
            [1, 4, 3, 2, 0],
            [1, 1, 0, 1, 0],
            [2, 3, 2, 1, 1],
            [0, 2, 0, 1, 2],
        ],
        "kernel": [
            [0, 1, 0, 1],
            [1, 0, 2, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        "input": [
            [1, 0, 1, 0, 3],
            [2, 0, 2, 1, 1],
            [3, 1, 3, 1, 0],
            [0, 3, 2, 0, 0],
            [1, 0, 2, 3, 1],
        ],
        "kernel": [
            [2, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 3, 0, 0],
        ],
    },
    {
        "input": [
            [0, 0, 1, 0, 2],
            [1, 2, 2, 1, 0],
            [1, 1, 0, 3, 1],
            [0, 3, 1, 0, 0],
            [1, 0, 2, 0, 2],
        ],
        "kernel": [
            [3, 0, 1, 0],
            [0, 2, 0, 1],
            [0, 1, 1, 0],
        ],
    },
]


def valid_products(image: np.ndarray, kernel: np.ndarray, *, flip: bool = False):
    """Return each valid patch product and its sum."""
    if flip:
        kernel = np.flip(kernel)
    out_rows = image.shape[0] - kernel.shape[0] + 1
    out_cols = image.shape[1] - kernel.shape[1] + 1
    products = []
    output = np.zeros((out_rows, out_cols), dtype=int)
    for row in range(out_rows):
        row_products = []
        for col in range(out_cols):
            product = image[
                row : row + kernel.shape[0], col : col + kernel.shape[1]
            ] * kernel
            row_products.append(product)
            output[row, col] = int(product.sum())
        products.append(row_products)
    return products, output


def show_question(number: int, image: np.ndarray, kernel: np.ndarray, *, flip: bool):
    convention = "mathematical convolution (flipped kernel)" if flip else "cross-correlation (CNN convention)"
    products, output = valid_products(image, kernel, flip=flip)
    print(f"Question {number}: {convention}")
    print("Effective kernel:\n", np.flip(kernel) if flip else kernel)
    for row, row_products in enumerate(products):
        for col, product in enumerate(row_products):
            print(f"  output[{row}, {col}] products:\n{product}")
            print(f"  output[{row}, {col}] sum = {product.sum()}")
    print("Output matrix:\n", output)
    print()
    return output


def main() -> None:
    for number, question in enumerate(QUESTIONS, start=1):
        image = np.asarray(question["input"], dtype=int)
        kernel = np.asarray(question["kernel"], dtype=int)
        print("Input:\n", image)
        print("Kernel:\n", kernel)
        show_question(number, image, kernel, flip=False)
        show_question(number, image, kernel, flip=True)


if __name__ == "__main__":
    main()
