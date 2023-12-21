import random
import numpy as np
from scipy.stats import shapiro, norm

def generate_integers():
    """Generate 4 random integers between 1 and 40."""
    return [random.randint(1, 40) for _ in range(4)]

def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    return sum(numbers)
	
from scipy.stats import boxcox
import numpy as np

def transform_to_normal(data):
    """
    Check if the distribution is normal, and if not, apply Box-Cox transformation.

    Parameters:
    - data: Input data array.

    Returns:
    - transformed_data: Transformed data array.
    - transformation_matrix: Matrix of transformations for retrieval of original values.
    """
    _, p_value = shapiro(data)
    print(f"Original Data Shapiro-Wilk p-value: {p_value}")

    if p_value < 0.05:  # If not normal, apply Box-Cox transformation
        transformed_data, transformation_matrix = boxcox(data)
    else:
        transformed_data = data
        transformation_matrix = None  # No transformation applied

    return transformed_data, transformation_matrix

# Example usage:
# Assuming you have a dataset that might not be normal
non_normal_data = np.random.exponential(size=100)

# Apply transformation
transformed_data, transformation_matrix = transform_to_normal(non_normal_data)

# Check if the distribution is now normal
_, p_value_after_transformation = shapiro(transformed_data)


print(f"Transformed Data Shapiro-Wilk p-value: {p_value_after_transformation}")


def calculate_distribution_statistics(sums):
    """
    Calculate distribution statistics including mean, standard deviation,
    and check for normality using Shapiro-Wilk test.

    Parameters:
    - sums: List of sum values.

    Returns:
    - mean: Mean of the sums.
    - std_dev: Standard deviation of the sums.
    - is_normal: True if the distribution is normal, False otherwise.
    """
    mean = np.mean(sums)
    std_dev = np.std(sums)
    _, p_value = shapiro(sums)
    is_normal = p_value > 0.05  # If p-value > 0.05, assume normal distribution

    return mean, std_dev, is_normal

def calculate_probability_of_sum(target_sum, means, std_devs):
    """
    Calculate the probability of a given sum in a normal distribution.

    Parameters:
    - target_sum: The sum for which probability needs to be calculated.
    - means: Mean values for different distributions.
    - std_devs: Standard deviation values for different distributions.

    Returns:
    - probability: Probability of the given sum in the normal distribution.
    """
    z_score = (target_sum - means) / std_devs
    probability = norm.cdf(z_score)

    return probability

def calculate_transition_probabilities(sums):
    """
    Calculate the transition probabilities (lambda values) from one sum to the next.

    Parameters:
    - sums: List of sum values.

    Returns:
    - lambdas: Array of transition probabilities.
    """
    lambdas = []

    for i in range(len(sums) - 1):
        current_sum = sums[i]
        next_sum = sums[i + 1]

        probability_current = calculate_probability_of_sum(current_sum, mean, std_dev)
        probability_next = calculate_probability_of_sum(next_sum, mean, std_dev)

        if probability_current == 0:
            # Avoid division by zero
            lambda_value = 0
        else:
            lambda_value = probability_next/(probability_current*(1-probability_current))

        lambdas.append(lambda_value)

    return np.array(lambdas)

############################################
##### !!! ML MODEL WITH TENSORFLOW !!! #####
############################################

#pip install tensorflow
	
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
    input_shape = X_train.shape[1:]
    #print(input_shape)
    model = create_lstm_model(input_shape=input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_next_lambda(model, X):
    ## Reshape input for prediction
    ##print(X)
    ##X = np.reshape(X, (1, X.shape[0], X.shape[1]))
    X = [[[X]]]
    # Make prediction
    predicted_lambda = model.predict(X, verbose=0)
    return predicted_lambda[0, 0]


# Example usage:
# Assuming you have a dataset of historic lambdas
historic_lambdas = np.array([0.8, 1.2, 0.9, 1.1])

# Prepare data for training
X_train = historic_lambdas[:-1]
y_train = historic_lambdas[1:]

#print(X_train)
#print(y_train)

# Reshape input for training
X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))

#print(X_train)

# Train the LSTM model
model = train_lstm_model(X_train, y_train, epochs=100, batch_size=1)

# Predict the next lambda value
predicted_lambda = predict_next_lambda(model, historic_lambdas[-1])
print("Predicted Next Lambda:", predicted_lambda)

############################################
##### !!! ML MODEL WITH TENSORFLOW !!! #####
############################################

def calculate_sum_from_probability(probability, means, std_devs):
    """
    Calculate the sum or sums given a probability.

    Parameters:
    - probability: Probability for which the sum(s) needs to be calculated.
    - means: Mean values for different distributions.
    - std_devs: Standard deviation values for different distributions.

    Returns:
    - calculated_sums: List of calculated sum(s) corresponding to the given probability.
    """
    possible_sums = np.arange(1, 41)

    # Calculate the cumulative distribution function (CDF) for each sum
    cdf_values = [calculate_probability_of_sum(s, means, std_devs) for s in possible_sums]

    # Find the index where the CDF exceeds the given probability
    index = np.argmax(np.array(cdf_values) >= probability)

    # Calculate the sum or sums corresponding to the given probability
    calculated_sums = possible_sums[:index + 1].tolist()

    return calculated_sums

# Example usage:
target_probability = 0.8  # Replace with the actual target probability
mean = 21
std_dev = 10

calculated_sums = calculate_sum_from_probability(target_probability, mean, std_dev)
print(f"Calculated Sum(s) for Probability {target_probability}: {calculated_sums}")



def predict_next_sum(predicted_lambda, current_sum, means, std_devs):
    """
    Predict the next sum based on the predicted lambda value.

    Parameters:
    - predicted_lambda: Predicted lambda value for the next transition.
    - current_sum: Current sum value.
    - means: Mean values for different distributions.
    - std_devs: Standard deviation values for different distributions.

    Returns:
    - predicted_sum: Predicted sum value for the next transition.
    """
    # Calculate the probability of the current sum
    probability_current = calculate_probability_of_sum(current_sum, means, std_devs)

    # Calculate the probability of the next possible sums
    #next_possible_sums = np.arange(1, 41)
    #probabilities_next = [probability_current * predicted_lambda * calculate_probability_of_sum(s, means, std_devs) for s in next_possible_sums]
    probabilities_next = probability_current * predicted_lambda * (1-probability_current)

    # Normalize probabilities to make sure they sum to 1
    #probabilities_next /= sum(probabilities_next)

    # Choose the next sum based on probabilities
    #predicted_sum = np.random.choice(next_possible_sums, p=probabilities_next)
    predicted_sum = calculate_sum_from_probability(probabilities_next,means,std_devs)
	
    return predicted_sum

# Example usage:
predicted_lambda = 1.2  # Replace with the actual predicted lambda value
current_sum = 120  # Replace with the actual current sum value

predicted_sum = predict_next_sum(predicted_lambda, current_sum, mean, std_dev)
print("Predicted Next Sum:", predicted_sum)



# Example usage:
random_sums = [calculate_sum(generate_integers()) for _ in range(1000)]

mean, std_dev, is_normal = calculate_distribution_statistics(random_sums)
print(f"Mean: {mean}, Standard Deviation: {std_dev}, Normal Distribution: {is_normal}")

target_sum = 120
probability = calculate_probability_of_sum(target_sum, mean, std_dev)
print(f"Probability of getting a sum of {target_sum}: {probability}")

transition_probabilities = calculate_transition_probabilities(random_sums)
print("Transition Probabilities (Lambda values):", transition_probabilities)
