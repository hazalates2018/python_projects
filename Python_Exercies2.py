# Implement a function to perform binary search ona sorted list. Use the list [1,3,5,7,9,11,13,15] and search for the element 7
def binary_search(arr,target):
    left, right = 0,len(arr) -1
    while left <=right:
        mid = (left + right) //2
        if arr[mid] == target:
            return mid
        elif arr[mid]<target:
            left = mid + 1
        else:
            right = mid -1
    return-1
arr = [1,3,5,7,9,11,13,15]
target = 7
index = binary_search(arr,target)
print(index)



# Write a function to solve the knapsack problem using dynamic programming. Use values = [60, 100, 120] weights = [10, 20, 30] and capacity 50
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1 ) for _ in range(n + 1)]
    for i in range(1, n+ 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w],values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]

weights = [10,20,30]
values = [60,100,120]
capacity = 50

print(knapsack(weights, values, capacity)) 





# implement a function to find the k'th smallest element in a list using the quickselect algorithm. Demonstrate with the list [7, 10, 4, 3, 20, 15] and k=3

import random

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)  # Choose a random pivot
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    if k <= len(left):  # Search in left partition
        return quickselect(left, k)
    elif k <= len(left) + len(mid):  # Pivot is the k-th smallest
        return pivot
    else:  # Search in right partition
        return quickselect(right, k - len(left) - len(mid))

arr = [7,10,4,3,20,15]
k = 3  # Find 3rd smallest element
print(quickselect(arr, k))  # Output: 6




# Write a Python class named 'Node' that represents a node in a singly linked list. Each node should have two attributes: 'data' (to store the value)
#  and 'next' (to point to the next node). Then, create three nodes with values 10, 20, and 30 and link them together. Finally, traverse the list and print each node's data.
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

node1 = Node(10)
node2 = Node(20)
node3 = Node(30)

node1.next = node2
node2.next = node3

current_node = node1
while current_node:
    print(current_node.data)
    current_node = current_node.next




# Create a Python program that defines a dictionary 'student_grades' with keys as student names ("John", "Alice", "Bob") and values as their grades (85, 90, 78).
# Then, iterate over the dictionary and print each student's name and grade in the format "John has a grade of 85."

student_grades = {"John":"85","Alice":"90","Bob":"78"}
for student,grade in student_grades.items():
    print(student+" has a grade of "+grade)
 


# Implement a Python class named 'Stack' using a list to represent the stack. The class should include methods to 'push' (add an element), 'pop' (remove the top element),
# and 'peek' (view the top element without removing it). Demonstrate the stack operations by pushing the numbers 5, 10, and 15 onto the stack, then popping one element 
# and printing the current top of the stack.

class Stack:
    def __init__(self):
        self.stack = []  

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return 

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return 

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

stack = Stack()

stack.push(5)
stack.push(10)
stack.push(15)

popped_element = stack.pop()
print(f"Popped Element: {popped_element}")  

top_element = stack.peek()
print(f"Top Element: {top_element}")  

# Create a variable name and assign your name to it. Then, print "Hello, [name]!" to the console.
name = "Hazal"
print(f"Hello",name) 


# Write a Python program to add two numbers and print the result.
def aggregation(num1,num2):
    return num1+num2

print(aggregation(8,10))


# Write a Python program to calculate the square of the number 7.

def square(num,root):
    return num**root
    
print(square(7,2))


num = 7
square = num**2
print(square)


# Create a for loop that prints numbers from 1 to 5.
for i in range(1,6):
    print(i)


# Write a Python program that takes a string "Python Programming" and stores it in a variable.
# Then, find and print the length of the string. Finally, print a message that says "The length of the string is [length] characters."
my_string = "Python Programming"
length = len(my_string)
print(f"The length of the string is {length} characters")


# Create a function named 'greet' that takes a name as an argument and prints "Hello, [name]!"
def greet(name):
    return print(f"Hello "+name+"!")

greet("Hazal")


# Write a Python program that checks if the word "orange" is in the list ["apple", "banana", "cherry", "date"].
#  If the word is found, print "The word is in the list!", otherwise print "The word is not in the list."

Fruits = ["apple", "banana", "cherry", "date"]

if "orange" in Fruits:
    print("The word is in the list")
else:
    print("The word is not in the list.") 



# Write a program that checks if a number is even or odd.

def check_num(num):
    if num % 2 ==0:
        print(f"{num} is even")
    else:
        print(f"{num} is odd")

check_num(20)


# Create a list of three fruits (apple, banana, cherry) and print the second fruit in the list.

fruits = ["apple", "banana", "cherry"]

print(fruits[1])



# Write a Python program that takes a list of integers, numbers = [2, 4, 6, 8, 10], and 
# calculates the sum of all the elements in the list. Then, print a message that says "The sum of the numbers is [sum]."

numbers = [2, 4, 6, 8, 10]
# total = sum(numbers)
# print(total)
total=0
for num in numbers:
    total += num
print(total)


# Write a Python function named 'is_prime' that takes an integer as an argument and 
# returns True if the number is prime and False if it is not. Test the function with the number 7 and print the result.

def is_prime(integ):
    if integ <=1:
        return False
    for i in range(2,integ):
        if integ % i ==0:
            return False
    return True

result = is_prime(7)

print(result)




# Create a Python program that takes a list of strings ["apple", "banana", "cherry", "date"] 
# and prints only the strings that start with the letter "b".

strings = ["apple", "banana", "cherry", "date"]
#filtered_strings = [s for s in strings if s.startswith("b")]
#print(filtered_strings)


for string in strings:
    if string.startswith("b"):
        print(string)



# Write a Python function named 'factorial' that takes a positive integer 'n' as an argument 
# and returns its factorial using a 'for' loop. Then, call the function with the value '5' and print the result.
def factorial(n):
    result = 1
    for i in range(1,n+1):
        result *=i
    return result

print(factorial(5))




# Write a Python program that checks if a given year, say '2024', is a leap year.
# A year is a leap year if it is divisible by '4' but not by '100', except when it is divisible by '400'. 
# Print "2024 is a leap year." or "2024 is not a leap year." based on the result.

def year(n):
    if (n % 4 == 0 and n % 100 !=0) or (n % 400 ==0):
        print(f"{n} is a leap year")
    else:
        print(f"{n} is not a leap year.")

year(2024)



# Create a Python function named 'reverse_string' that takes a string as an argument and returns the string in reverse order.
# Test the function with the string "Python" and print the result.


def reverse_string(n):
    print(n[::-1])


reverse_string("Python")


# Write a Python program that defines a list of integers and removes all the duplicate elements from the list.
# Then, print the list with duplicates removed. For example, if 'numbers = [1, 2, 2, 3, 4, 4, 5]', the result should be '[1, 2, 3, 4, 5]'.


numbers = [1, 2, 2, 3, 4, 4, 5]
numbers = list(set(numbers))
print(numbers)



# Write named a Python function 'count_vowels' that takes a string as an argument and returns the number of vowels (a, e, i, o, u) in the string.
# Test the function with the string "Hello World" and print the result.



def count_vowels(n):
    vowels = "aeiouAEIOU"
    count = 0
    for char in n:
        if char in vowels:
            count +=1
    return count
result = count_vowels("Hello World")

print(result)


# Write a Python function that takes a string as input and returns the string reversed.

def düz(n):
    return n[::-1]

print(düz("kedi"))




# Write a function that finds and returns the largest number in a given list.

def largest(liste):
    return max(liste)

print(largest([3,8,1,122,23,456,678,0,7,56,7]))



# Write a function that checks whether a given number is prime.

def prime_detector(n):
    if n <2:
        return False
    for i in range(2, int(n ** 0.5) +1):
        if n % i ==0:
            return False
    return True

print(prime_detector(29))



# Write a Python function that counts the number of words in a given sentence.

def words_counter(n):
    return len(n.split())

print(words_counter("Selam canım naber iyi canım senden naber"))




# Write a function that checks if a string reads the same forward and backward.

def is_palindrome(n):
    return n == n[::-1]

print(is_palindrome("racecar"))



# Write a function that calculates the factorial of a given number.
def factorial(n):
    result = 1
    for i in range(1,n + 1):
        result *= i
    return result

print(factorial(5))



# Write a function that generates the Fibonacci series up to n terms.
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b

print(fibonacci(7))

















