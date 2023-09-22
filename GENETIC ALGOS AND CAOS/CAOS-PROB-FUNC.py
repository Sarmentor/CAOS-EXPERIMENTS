#Please code a function that given the probability of ocurrence of a normal distributed function retrieves its probability with caos theory formula
#Certainly! Here's a Python function that calculates the probability of occurrence of a normally distributed function using the chaos theory formula:

#```python
import math

def chaos_theory_probability(probability):
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1.")

    z_score = math.sqrt(2) * math.erfinv(2 * probability - 1)

    return 1 - (1 / (1 + math.exp(z_score)))

# Example usage
probability = 0.8
chaos_probability = chaos_theory_probability(probability)
print("Chaos theory probability:", chaos_probability)
#```

#In this code, we first check if the given probability is within the valid range of 0 to 1. If not, we raise a `ValueError` indicating the invalid input. 

#Then, we use the inverse error function (`erfinv`) from the `math` module to calculate the z-score corresponding to the given probability. The z-score is a measure of how many standard deviations away from the mean a specific value is in a normal distribution.

#Finally, we calculate the chaos theory probability using the formula `1 - (1 / (1 + exp(z_score)))`, where `exp` is the exponential function.

#Note that you'll need to have the `math` module imported in your Python environment to use this function.