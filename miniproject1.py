"""
Filename:     miniproject1.py
Version:      1.0
Date:         2018/2/7

Description:  Experimentation with simple prediction algorithms to be used for
              CS 155's first miniproject.

Author(s):     Garret Sullivan, Dennis Lam
Organization:  California Institute of Technology

"""

# Import package components
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

# Import packages as aliases
import matplotlib.pyplot as plt
import numpy as np


def parse_words(line, return_list=True):
    """
    Returns the bag of words in a list or dictionary.

    """
    if return_list:
        return [word for word in line.split(" ")]
    else:
        return {word:0 for word in line.split(" ")}


def parse_training_data(line):
    """
    Parses a line from the file containing the training data.

    """
    input_vector = [int(x) for x in line.split(" ")]
    target_variable = input_vector.pop(0)
    
    return (input_vector, target_variable)


def parse_test_data(line):
    """
    Parses a line from the file containing the test data.

    """
    return [int(x) for x in line.split(" ")]


def assemble_training_dataset(input_filename):
    """
    Given the name of the file containing the training data, returns a triple
    containing a list of the words, a list of the input vectors, and a list of
    the target values.

    """
    # Stores the training feature vectors and target values
    X = []
    Y = []
    
    with open(input_filename) as input_file:
        # Store the words in a list
        word_list = parse_words(input_file.readline())

        # Process the rest of the training data
        for line in input_file:
            x_temp, y_temp = parse_training_data(line)

            # Add the data from the current line to the lists of input vectors
            # and target vectors
            X.append(x_temp)
            Y.append(y_temp)

    # Remove the leading "Label" string at the beginning of the list of words
    word_list.pop(0)

    # Remove the trailing end-of-line character from the last word
    word_list[len(word_list) - 1] = word_list[len(word_list) - 1].rstrip()

    # Convert the lists of input vectors and target vectors to NumPy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Return the complete training dataset
    return (word_list, X, Y)


def assemble_test_dataset(input_filename):
    """
    Given the name of the file containing the test data, returns a tuple
    containing a list of the words and a list of the input vectors.

    """
    # Stores the test feature vectors
    X = []
    
    with open(input_filename) as input_file:
        # Store the words in a list
        word_list = parse_words(input_file.readline())

        # Process the rest of the test data
        for line in input_file:
            # Add the data from the current line to the lists of input vectors
            X.append(parse_test_data(line))

    # Remove the trailing end-of-line character from the last word
    word_list[len(word_list) - 1] = word_list[len(word_list) - 1].rstrip()

    # Convert the list of input vectors to a NumPy array
    X = np.asarray(X)
    
    # Return the complete training dataset
    return (word_list, X)


def make_submission_file(output_filename, predictions):
    """
    Generates a properly formatted submission file containing the given
    predictions and with the given filename.

    """
    with open(output_filename, "w+") as output_file:
        # Write the column labels
        output_file.write("Id,Prediction\n")

        # Initialize the first prediction ID number to 1
        id = 1

        # Write the predictions to the file
        for x in predictions:
            output_file.write(str(id) + "," + str(x) + "\n")
            id += 1


def main():

    # The location and names of the data files
    input_directory = "data"
    training_filename = "training_data.txt"
    test_filename = "test_data.txt"

    # Read, parse, and store the training data
    training_word_list, training_data, training_values = assemble_training_dataset(input_directory + "\\" + training_filename)

    # Read, parse, and store the test data
    test_word_list, test_data = assemble_test_dataset(input_directory + "\\" + test_filename)

    # Combine the data into a single array
    all_data = np.vstack((training_data, test_data))
    
    # Apply the TF-IDF transformation to the data
    transformer = TfidfTransformer()
    transformed_data = transformer.fit_transform(all_data) 
    
    # Split the data back into training and test sets
    split_data = np.array_split(transformed_data.toarray(), [len(training_data)])
    X_train = split_data[0]
    X_test = split_data[1]

    ###########################################################################
    #                                                                         #
    # Test the cross-validation accuracies of different regularization levels #
    #                                                                         #
    ###########################################################################

    # Stores cross-validation scores
    cv_scores = []

    # Stores the regularization levels
    reg_values = []

    # The number of folds to use for cross-validation
    num_folds = 5
    
    # Test different levels of regularization
    for i in range(1, 50 + 1):
        # Set the regularization parameter
        reg_value = i * 0.1

        # Define the model
        model = LogisticRegression(C=reg_value)

        # Obtain the average cross-validation score
        cv_score = np.mean(cross_val_score(model, X_train, training_values, cv=num_folds, n_jobs=1))

        # Print the results to standard output
        print("[C = " + str(reg_value) + "]")
        print(cv_score)

        # Save the cross-validation score
        cv_scores.append(cv_score)
    
        # Save the regularization parameter
        reg_values.append(reg_value)

    # Convert the lists to NumPy arrays
    cv_scores = np.asarray(cv_scores)
    reg_values = np.asarray(reg_values)

    # Plot the average cross-validation scores as a function of the regularization parameter
    plt.plot(reg_values, cv_scores)

    # Label the plot
    plt.title("Cross-Validation Accuracy as a Function of Regularization")
    plt.xlabel("Value of Regularization Parameter")
    plt.ylabel("Average " + str(num_folds) + "-fold Cross-Validation Accuracy")

    # Define the plot's boundaries
    plt.xlim(reg_values[0], reg_values[len(reg_values) - 1])
    plt.ylim(0.840, 0.855)

    # Save the plot to file
    plt.savefig("cross-validation_accuracies.png", dpi=200)

    # Display the plot
    plt.show()

    ##########################################
    #                                        #
    # Train a model and generate predictions #
    #                                        #
    ##########################################
    
    # The location and names of the output files
    output_directory = "output"
    output_filename = "submission_TF-IDF_C1.8.txt"

    # Define the model
    model = LogisticRegression(C=1.8)

    # Train the model
    model.fit(X_train, training_values)
    
    # Generate predictions
    predictions = model.predict(X_test)

    # Write the predictions to a file
    make_submission_file(output_directory + "\\" + output_filename, predictions)


if __name__ == "__main__":
    main()
