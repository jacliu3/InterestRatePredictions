friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print "name %i is %s" % (i, name)   #changed %i to %s for printing names

# How many friends contain the letter 'a' ?
count_a = 0.0             #changed to double to prevent int roudning
for name in friends:
    if 'a' in name:       #changed a to a string instead of undeclared variable
        count_a += 1      #python is silly and doesn't have increment operators     

print "%2.1f percent of the names contain an 'a'" % ( count_a / len(friends) * 100)
#changed formatting of percent so
#a) it is a percent, not a decimal
#b) rounds to one decimal

# Say hi to all friends
def print_hi(name, greeting='hello'):   #default arguments go after non-default ones
    print "%s %s" % (greeting, name)

map(print_hi, friends)

# Print sorted names out
print sorted(friends)       #using friends.sort() will modify the original list
                            #and then not return anything

'''
Calculate the factorial N! = N * (N-1) * (N-2) * ...
(use triple quotes for multi-line comments)
'''

def factorial(x):
    """
    Calculate factorial of number
    :param N: Number to use
    :return: x!
    """
    if x==1: return 1
    elif x < 1: return None     #don't want to go into an infinite loop
    return factorial(x-1) * x   #gotta multiply by the number itself too

print "The value of 5! is", factorial(5)
