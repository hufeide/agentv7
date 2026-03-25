def multiply(a, b):
    """Multiply two numbers and return the result.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
    
    Returns:
        The product of a and b
    
    Examples:
        >>> multiply(2, 3)
        6
        >>> multiply(2.5, 4)
        10.0
    """
    return a * b


if __name__ == "__main__":
    # Test examples
    print(multiply(2, 3))      # Output: 6
    print(multiply(2.5, 4))    # Output: 10.0
    print(multiply(-1, 5))     # Output: -5
