"""Sequential Minimal Optimization used to figure out the optimal separating hyperplane"""
import random

def select_alpha2(alpha1, num_alphas):
    """a random value is chosen for alpha2 as long as it does not equal alpha1

    :param alpha1: value which alpha2 cannot take
    :param num_alphas: total number of alphas
    :return: integer that will be the value for alpha2
    """
    alpha2 = alpha1
    while(alpha2==alpha1):
        alpha2 = random.uniform(0, num_alphas)

    return alpha2