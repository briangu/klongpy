import math

def cosine_distance(a, b):
    dot_product = sum(ai * bi for ai, bi in zip(a, b))
    magnitude_a = math.sqrt(sum(ai * ai for ai in a))
    magnitude_b = math.sqrt(sum(bi * bi for bi in b))
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    print(dot_product, magnitude_a, magnitude_b, cosine_similarity)
    return 1 - cosine_similarity

a = [0.2242275924404239, 0.882144226374053, 0.9906177571863527, 0.4108451091093793, 0.9804472187642895, 0.4060536401874264, 0.3002237000518012, 0.8514871407116807, 0.6532517687913413, 0.8627410006208077]
b = [0.30545495121740496, 0.3771115787379501, 0.7078223604498113, 0.019355013585787284, 0.16809292508968254, 0.20411410383993234, 0.09404528697075443, 0.8849676641314673, 0.9028064826465457, 0.8102216701662028]

print(cosine_distance(a,b))