# Python Programming Basics

## Introduction to Python
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant use of whitespace indentation.

## Data Types
Python supports several built-in data types:

### Numbers
- **Integers**: Whole numbers like 1, 42, -17
- **Floats**: Decimal numbers like 3.14, -0.5, 2.718
- **Complex**: Numbers with real and imaginary parts like 3+4j

### Strings
Strings are sequences of characters enclosed in quotes. You can use single quotes ('hello') or double quotes ("world").

### Lists
Lists are ordered, mutable collections defined with square brackets:
```python
my_list = [1, 2, 3, "apple", "banana"]
my_list.append("cherry")  # Add element
my_list[0] = 10  # Modify element
```

### Dictionaries
Dictionaries are key-value pairs defined with curly braces:
```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}
print(person["name"])  # Output: Alice
```

## Control Flow

### Conditional Statements
```python
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")
```

### Loops
For loops iterate over sequences:
```python
for i in range(5):
    print(i)  # Prints 0, 1, 2, 3, 4

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

While loops continue until a condition is false:
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

## Functions
Functions are defined using the `def` keyword:
```python
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

def add(a, b):
    """Add two numbers."""
    return a + b

result = add(5, 3)  # result = 8
message = greet("Alice")  # message = "Hello, Alice!"
```

Functions can have default parameters:
```python
def power(base, exponent=2):
    """Calculate power with default exponent of 2."""
    return base ** exponent

print(power(3))     # 9 (3^2)
print(power(3, 3))  # 27 (3^3)
```

## Classes and Objects
Python supports object-oriented programming:
```python
class Dog:
    """A simple dog class."""

    def __init__(self, name, age):
        """Initialize dog with name and age."""
        self.name = name
        self.age = age

    def bark(self):
        """Make the dog bark."""
        return f"{self.name} says Woof!"

    def get_age_in_dog_years(self):
        """Calculate age in dog years (7x human years)."""
        return self.age * 7

my_dog = Dog("Buddy", 3)
print(my_dog.bark())  # Output: Buddy says Woof!
print(my_dog.get_age_in_dog_years())  # Output: 21
```

## Exception Handling
Handle errors gracefully with try-except:
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always executes")
```

## File Operations
Reading and writing files:
```python
# Writing to a file
with open("output.txt", "w") as file:
    file.write("Hello, World!")

# Reading from a file
with open("input.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

## List Comprehensions
Concise way to create lists:
```python
# Traditional way
squares = []
for i in range(10):
    squares.append(i ** 2)

# List comprehension
squares = [i ** 2 for i in range(10)]

# With condition
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
```

## Lambda Functions
Anonymous functions for simple operations:
```python
# Regular function
def double(x):
    return x * 2

# Lambda equivalent
double = lambda x: x * 2

# Common use with map
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
```

## Key Takeaways
- Python uses indentation for code blocks
- Variables don't need type declarations
- Everything in Python is an object
- Use descriptive variable names for readability
- Follow PEP 8 style guidelines
- Use built-in functions and libraries when possible
