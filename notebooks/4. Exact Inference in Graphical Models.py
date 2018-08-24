
# coding: utf-8

# ### Inference
# Inference is same as asking conditional probability questions to the models. So in our student example we might would have liked to know what is the probability of a student getting a good grade given that he is intelligent which is basically equivalent of asking $ P(g^1 | i^1) $. Inference algorithms deals with efficiently finding these conditional probability queries.
# 
# There are two main categories for inference algorithms:
# 1. Exact Inference: These algorithms find the exact probability values for our queries.
# 2. Approximate Inference: These algorithms try to find approximate values by saving on computation.

# ### Exact Inference
# There are multiple algorithms for doing exact inference. We will mainly be talking about two very common algorithms in this notebook:
# 1. Variable Elimination
# 2. Clique Tree Belief Propagation

# ### Variable Elimination
# The basic concept of variable elimination is same as doing marginalization over Joint Distribution. But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors. So basically if we want to eliminate $ X $ from our distribution, then we compute the product of all the factors involving $ X $ and marginalize over them, thus allowing us to work on much smaller factors. Let's take the student example to make things more clear:
# 
# $$ P(D) = \sum_I \sum_S \sum_G \sum_L P(D, I, S, G, L) $$
# $$ P(D) = \sum_I \sum_S \sum_G \sum_L P(D) * P(I) * P(S | I) * P(G | D, I) * P(L | G) $$
# $$ P(D) = P(D) \sum_S P(S | I) \sum_I P(I) \sum_G P(G | D, I) \sum_L P(L | G) $$
# 
# In the above equation we can see that we pushed the summation inside and operated the summation only factors that involved that variable and hence avoiding computing the complete joint distribution.
# 
# Let's now see some code examples:

# In[ ]:


## Add code examples for Variable Elimination


# ### Clique Tree Belief Propagation
