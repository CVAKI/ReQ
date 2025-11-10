#!/usr/bin/env python3
"""
reQ Programming Language - Example Programs and Test Suite
Demonstrates various features of reQ
"""

# ============================================================================
# EXAMPLE 1: HELLO WORLD
# ============================================================================

HELLO_WORLD = """
{[MAIN: 1]}-${
    see("Hello, World!")
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 2: FACTORIAL (Tail Recursive)
# ============================================================================

FACTORIAL = """
{[MAIN: 1]}-${
    # Tail-recursive factorial with accumulator
    (: factorial_tail(n, accumulator)$
        if@ n <= 1$
            {[RETURN: (accumulator)]}
        else$
            {[RETURN: (factorial_tail(n - 1, n * accumulator))]}
    
    # Calculate factorial of 5
    result = factorial_tail(5, 1)
    see("Factorial of 5:", result)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 3: FIBONACCI (with Memoization)
# ============================================================================

FIBONACCI = """
{[MAIN: 1]}-${
    # Initialize memoization array
    memo = 1D[100]
    fill_array(memo, -1)
    
    (: fibonacci_memo(n, memo)$
        if@ n <= 1$
            {[RETURN: (n)]}
        elfi@ memo[n] != -1$
            {[RETURN: (memo[n])]}
        else$
            result = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
            memo[n] = result
            {[RETURN: (result)]}
    
    # Calculate Fibonacci(20)
    result = fibonacci_memo(20, memo)
    see("Fibonacci(20):", result)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 4: ARRAY PROCESSING WITH RECURSION
# ============================================================================

ARRAY_PROCESSING = """
{[MAIN: 1]}-${
    # Create and fill array
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Recursive sum
    (: sum_array(arr, index, accumulator)$
        if@ index >= length(arr)$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator + arr[index]
            {[RETURN: (sum_array(arr, index + 1, new_acc))]}
    
    # Recursive product
    (: product_array(arr, index, accumulator)$
        if@ index >= length(arr)$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator * arr[index]
            {[RETURN: (product_array(arr, index + 1, new_acc))]}
    
    total = sum_array(numbers, 0, 0)
    product = product_array(numbers, 0, 1)
    
    see("Sum:", total)
    see("Product:", product)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 5: MULTITHREADING
# ============================================================================

MULTITHREADING = """
{[THREAD: 1]}-${
    # Process first half of array
    (: process_chunk(data, start, end, index, accumulator)$
        if@ index >= end$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator + data[index] * data[index]
            {[RETURN: (process_chunk(data, start, end, index + 1, new_acc))]}
    
    result = process_chunk(dataset, 0, 500, 0, 0)
    {[RETURN: (result)]}
}

{[THREAD: 2]}-${
    # Process second half of array
    (: process_chunk(data, start, end, index, accumulator)$
        if@ index >= end$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator + data[index] * data[index]
            {[RETURN: (process_chunk(data, start, end, index + 1, new_acc))]}
    
    result = process_chunk(dataset, 500, 1000, 500, 0)
    {[RETURN: (result)]}
}

{[MAIN: 1]}-${
    # Create dataset
    dataset = 1D[1000]
    fill_random(dataset, 1.0, 100.0)
    
    # Execute threads in parallel
    sum1 = {[THREAD: 1]}
    sum2 = {[THREAD: 2]}
    
    total = sum1 + sum2
    see("Sum of squares:", total)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 6: FRIEND SYSTEM - PYTHON INTEGRATION
# ============================================================================

FRIEND_PYTHON = """
{[MAIN: 1]}-${
    # Generate data in reQ
    data = [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.1]
    
    # Process with Python
    result = {FRIEND["py"]}-${
        import statistics
        
        # Access reQ data
        values = data_from_req
        
        # Calculate statistics
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        stdev_val = statistics.stdev(values)
        
        # Return as dictionary
        {[RETURN: ({
            'mean': mean_val,
            'median': median_val,
            'stdev': stdev_val
        })]}
    }
    
    see("Python Statistics:")
    see("Mean:", result['mean'])
    see("Median:", result['median'])
    see("StdDev:", result['stdev'])
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 7: FRIEND SYSTEM - C++ INTEGRATION
# ============================================================================

FRIEND_CPP = """
{[MAIN: 1]}-${
    # Use C++ for performance-critical computation
    size = 1000000
    
    result = {FRIEND["cpp"]}-${
        #include <vector>
        #include <numeric>
        #include <algorithm>
        
        std::vector<long long> numbers(size_from_req);
        std::iota(numbers.begin(), numbers.end(), 1);
        
        // Sum of squares
        long long sum = 0;
        for(const auto& n : numbers) {
            sum += n * n;
        }
        
        std::cout << sum;
        
        {[RETURN: (sum)]}
    }
    
    see("C++ computed sum of squares:", result)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 8: HYBRID COMPUTING PIPELINE
# ============================================================================

HYBRID_PIPELINE = """
{[MAIN: 1]}-${
    # Step 1: Generate data in reQ
    raw_data = 1D[100]
    fill_random(raw_data, 0.0, 100.0)
    
    # Step 2: Process with Python
    normalized = {FRIEND["py"]}-${
        import numpy as np
        
        data = np.array(data_from_req)
        normalized = (data - data.mean()) / data.std()
        
        {[RETURN: (normalized.tolist())]}
    }
    
    # Step 3: Heavy computation in C++
    processed = {FRIEND["cpp"]}-${
        #include <vector>
        #include <cmath>
        
        std::vector<double> result;
        for(double val : normalized_from_req) {
            double transformed = std::pow(val, 3) + 2 * std::pow(val, 2);
            result.push_back(transformed);
        }
        
        {[RETURN: (result)]}
    }
    
    # Step 4: Final aggregation in reQ
    (: sum_list(lst, idx, acc)$
        if@ idx >= length(lst)$
            {[RETURN: (acc)]}
        else$
            {[RETURN: (sum_list(lst, idx + 1, acc + lst[idx]))]}
    
    total = sum_list(processed, 0, 0.0)
    average = total / length(processed)
    
    see("Pipeline complete!")
    see("Average:", average)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 9: 4D ARRAY - RUBIK'S CUBE
# ============================================================================

RUBIKS_CUBE = """
{[MAIN: 1]}-${
    # Create a 3x3x3 Rubik's Cube with 6 faces
    rubik = 4D-HEXAGON[3][3][3][6]
    
    # Initialize to solved state
    initialize_solved_cube(rubik)
    
    see("Rubik's Cube initialized")
    see("Is solved:", is_solved(rubik))
    
    # Perform some moves
    move_cube(rubik, "R")
    move_cube(rubik, "U")
    move_cube(rubik, "R'")
    move_cube(rubik, "U'")
    
    see("After moves, is solved:", is_solved(rubik))
    
    # Solve the cube
    solution = solve_cube(rubik)
    see("Solution moves:", solution)
    see("Final state, is solved:", is_solved(rubik))
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# EXAMPLE 10: EXCEPTION HANDLING
# ============================================================================

EXCEPTION_HANDLING = """
{[MAIN: 1]}-${
    (: safe_divide(a, b)$
        try$
            if@ b == 0$
                {[RETURN: (0)]}
            else$
                result = a / b
                {[RETURN: (result)]}
        exept-> DivisionByZero$
            see("Cannot divide by zero!")
            {[RETURN: (0)]}
        exept-> error$
            see("Unknown error:", error)
            {[RETURN: (-1)]}
    
    # Test division
    result1 = safe_divide(10, 2)
    result2 = safe_divide(10, 0)
    result3 = safe_divide(15, 3)
    
    see("Results:", result1, result2, result3)
    
    # Test FRIEND error handling
    try$
        py_result = {FRIEND["py"]}-${
            # This might fail
            import non_existent_module
            {[RETURN: (1)]}
        }
        see("Success:", py_result)
    exept-> ForeignLanguageError$
        see("Python module not found - using default")
        py_result = 0
    
    see("Final result:", py_result)
    
    {[RETURN: (nun)]}
}
"""

# ============================================================================
# TEST SUITE
# ============================================================================

class ReQTestSuite:
    """
    Test suite for reQ compiler
    """
    
    def __init__(self):
        self.examples = {
            'hello_world': HELLO_WORLD,
            'factorial': FACTORIAL,
            'fibonacci': FIBONACCI,
            'array_processing': ARRAY_PROCESSING,
            'multithreading': MULTITHREADING,
            'friend_python': FRIEND_PYTHON,
            'friend_cpp': FRIEND_CPP,
            'hybrid_pipeline': HYBRID_PIPELINE,
            'rubiks_cube': RUBIKS_CUBE,
            'exception_handling': EXCEPTION_HANDLING,
        }
    
    def list_examples(self):
        """
        List all available examples
        """
        print("📚 Available reQ Examples:")
        print("=" * 60)
        for i, name in enumerate(self.examples.keys(), 1):
            print(f"{i:2}. {name.replace('_', ' ').title()}")
        print()
    
    def get_example(self, name: str) -> str:
        """
        Get example code by name
        """
        return self.examples.get(name, "")
    
    def run_all_tests(self, compiler):
        """
        Run all test examples through the compiler
        """
        print("🧪 Running reQ Test Suite")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for name, code in self.examples.items():
            print(f"\n Testing: {name.replace('_', ' ').title()}")
            print("-" * 40)
            
            try:
                result = compiler.compile(code)
                if result:
                    print("✅ PASS")
                    passed += 1
                else:
                    print("❌ FAIL")
                    failed += 1
            except Exception as e:
                print(f"❌ FAIL: {str(e)}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"Test Results: {passed} passed, {failed} failed")
        print("=" * 60)

# ============================================================================
# BENCHMARK PROGRAMS
# ============================================================================

BENCHMARKS = {
    'recursive_sum': """
{[MAIN: 1]}-${
    (: recursive_sum(n, acc)$
        if@ n <= 0$
            {[RETURN: (acc)]}
        else$
            {[RETURN: (recursive_sum(n - 1, acc + n))]}
    
    result = recursive_sum(10000, 0)
    see("Sum:", result)
    {[RETURN: (nun)]}
}
""",
    
    'parallel_computation': """
{[THREAD: 1]}-${
    (: compute(n, acc)$
        if@ n <= 0$
            {[RETURN: (acc)]}
        else$
            {[RETURN: (compute(n - 1, acc + n * n))]}
    
    {[RETURN: (compute(5000, 0))]}
}

{[THREAD: 2]}-${
    (: compute(n, acc)$
        if@ n <= 0$
            {[RETURN: (acc)]}
        else$
            {[RETURN: (compute(n - 1, acc + n * n))]}
    
    {[RETURN: (compute(5000, 0))]}
}

{[MAIN: 1]}-${
    r1 = {[THREAD: 1]}
    r2 = {[THREAD: 2]}
    see("Total:", r1 + r2)
    {[RETURN: (nun)]}
}
"""
}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("reQ Programming Language - Examples & Test Suite")
    print("=" * 60)
    
    suite = ReQTestSuite()
    suite.list_examples()
    
    print("📝 Example Categories:")
    print("   • Basic: hello_world, factorial, fibonacci")
    print("   • Arrays: array_processing")
    print("   • Concurrency: multithreading")
    print("   • FRIEND System: friend_python, friend_cpp, hybrid_pipeline")
    print("   • Advanced: rubiks_cube, exception_handling")
    print()
    print("🚀 To compile an example:")
    print("   from beetroot_compiler import BeetrootCompiler")
    print("   compiler = BeetrootCompiler()")
    print("   compiler.compile(example_code)")
    print()
    print("💡 To run test suite:")
    print("   suite.run_all_tests(compiler)")