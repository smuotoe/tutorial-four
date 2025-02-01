import torch


def tensor_operations():
    """
    Instructions are provided as comments.
    """
    # Exercise 1: Creating Tensors
    # --------------------------
    # TODO: Create a 3x3 tensor filled with integers from 1 to 9
    # Hint: Use torch.tensor() and a nested list
    tensor_3x3 = None  # Replace None with your code

    # Exercise 2: Tensor Operations
    # --------------------------
    # TODO: Perform the following operations:
    # 1. Add 5 to every element in tensor_3x3
    # 2. Multiply every element by 2
    # Store results in new tensors
    added_tensor = None  # Replace None with your code
    multiplied_tensor = None  # Replace None with your code

    # Exercise 3: Matrix Operations
    # --------------------------
    # TODO: Create two 2x2 matrices and perform matrix multiplication
    # Matrix 1 should be [[2, 0], [0, 2]]
    # Matrix 2 should be [[1, 2], [3, 4]]
    matrix_1 = None  # Replace None with your code
    matrix_2 = None  # Replace None with your code
    result_matrix = None  # Replace None with your code

    # Exercise 4: Reshaping
    # --------------------------
    # TODO: Create a 1D tensor with numbers 1-8
    # Then reshape it into a 2x4 matrix
    # Hint: tensor.reshape
    tensor_1d = None  # Replace None with your code
    tensor_2x4 = None  # Replace None with your code

    # Exercise 5: Indexing and Slicing
    # --------------------------
    practice_tensor = torch.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12]])

    # TODO: Extract the following from practice_tensor:
    # 1. The second row
    # 2. The last column
    # 3. A 2x2 sub-matrix from the top-left corner
    second_row = None  # Replace None with your code
    last_column = None  # Replace None with your code
    sub_matrix = None  # Replace None with your code

    return {
        'tensor_3x3': tensor_3x3,
        'added_tensor': added_tensor,
        'multiplied_tensor': multiplied_tensor,
        'matrix_multiplication': result_matrix,
        'reshaped_tensor': tensor_2x4,
        'second_row': second_row,
        'last_column': last_column,
        'sub_matrix': sub_matrix
    }


if __name__ == "__main__":
    print(tensor_operations())
