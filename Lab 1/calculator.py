"""A simple calculator program with basic operations."""

def add(num1, num2):
    """Return the sum of two numbers."""
    return num1 + num2


def subtract(num1, num2):
    """Return the difference of two numbers."""
    return num1 - num2


def multiply(num1, num2):
    """Return the product of two numbers."""
    return num1 * num2


def divide(num1, num2):
    """Return the division of two numbers."""
    if num2 == 0:
        return "Error: Division by zero"
    return num1 / num2


def modulo(num1, num2):
    """Return the modulo of two numbers."""
    if num2 == 0:
        return "Error: Modulo by zero"
    return num1 % num2


def main():
    """Main function to run calculator."""
    print("Simple Calculator")
    print("Operations available: add, subtract, multiply, divide, modulo")

    while True:
        operation = input("Enter operation (or 'exit' to quit): ").strip().lower()
        if operation == "exit":
            print("Exiting calculator. Goodbye!")
            break

        try:
            number1 = float(input("Enter first number: "))
            number2 = float(input("Enter second number: "))
        except ValueError:
            print("Invalid input! Please enter numeric values.")
            continue

        if operation == "add":
            print("Result:", add(number1, number2))
        elif operation == "subtract":
            print("Result:", subtract(number1, number2))
        elif operation == "multiply":
            print("Result:", multiply(number1, number2))
        elif operation == "divide":
            print("Result:", divide(number1, number2))
        elif operation == "modulo":
            print("Result:", modulo(number1, number2))
        else:
            print("Unknown operation. Please try again.")


if __name__ == "__main__":
    main()
