# reQ Programming Language: Complete Specification v2.0
Table of Contents
Language Philosophy and Design GoalsCore Language StructureType System and Data TypesSyntax and SemanticsBlock Structure SystemArray Systems and Multi-Dimensional DataRecursion-Only Programming ModelHyper-Multithreading ArchitectureMulti-Language Embedding System (FRIEND)Memory ManagementException HandlingStandard LibraryCompiler ArchitectureDevelopment ToolsComplete Language Reference
1. Language Philosophy and Design Goals
1.1 Core Philosophy
reQ (recursive Queue) represents a revolutionary approach to high-performance programming built on four foundational principles:
Performance First: reQ compiles directly to optimized assembly language through the beetroot compiler, eliminating runtime interpretation overhead. Every language feature is designed with performance implications in mind.
Pure Recursion: Unlike traditional languages that offer both loops and recursion, reQ exclusively uses recursion for all repetitive operations. This enables automatic tail-call optimization, converting recursive calls into efficient loops at the assembly level.
Explicit Parallelism: With support for 100+ concurrent threads through hyper-multithreading, reQ makes parallelism a first-class citizen with automatic scheduling and work-stealing algorithms.
Polyglot Integration: The revolutionary FRIEND system allows seamless embedding of C++, Python, and Java code blocks, enabling developers to leverage existing ecosystems while maintaining reQ's performance benefits.
1.2 Design Goals
Near-C Performance: Within 5% of hand-optimized C codeMathematical Correctness: Native support for multi-dimensional arrays with compile-time validationDeveloper Productivity: Clear syntax with Python-like indentationSafe Concurrency: Automatic memory management prevents common threading bugsMulti-Language Integration: Zero-friction embedding of foreign code
1.3 Target Applications
Scientific computing and numerical simulations3D graphics and visualization systemsMachine learning pipelines (combining Python ML libraries with C++ performance)Game development with complex mathematical operationsHybrid applications requiring multiple language ecosystems
2. Core Language Structure
2.1 Program Organization
Every reQ program is organized into blocks with this fundamental structure:
req{[BLOCK_TYPE: IDENTIFIER]}-${
    # Block content
    # Statements and expressions
    {[RETURN: (value)]}
}

Block Types:
MAIN: Entry point of the programTHREAD: Concurrent execution unitFUNCTION: Optional grouping of related functions
2.2 Mandatory Return Policy
Every block must return a value. Even if logically returning "nothing," you must explicitly return nun (the null type). This ensures:
Clear data flow throughout the programCompiler verification of all code pathsExplicit thread return valuesNo implicit void functions
2.3 Indentation-Based Syntax
Following Python's philosophy, reQ uses indentation to define code blocks. The $ symbol marks the end of function headers and conditional statements.
Function Definition Syntax:
(: function_name(param1, param2)$
    # Function body with Python-like indentation
    result = param1 + param2
    {[RETURN: (result)]}

Key Points:
(: prefix starts function definitionParameters in parentheses$ terminates the function headerPython-style indentation for the bodyExplicit return with {[RETURN: (value)]}
3. Type System and Data Types
3.1 Primitive Types
Type Size Range/Description int 32-bit -2,147,483,648 to 2,147,483,647 float 64-bit IEEE 754 double precision char 1-4 bytes UTF-8 encoded Unicode string Variable UTF-8 immutable strings bool 1 byte true or false nun 0 bytes Null/void type
3.2 Array Types
One-Dimensional Arrays (1D):
numbers = 1D[10]              # Declaration
data = [1, 2, 3, 4, 5]        # Direct initialization

Two-Dimensional Arrays (2D):
matrix = 2D[3][4]             # 3 rows, 4 columns
grid = [[1, 2, 3], [4, 5, 6]] # Direct initialization

Three-Dimensional Arrays (3D):
volume = 3D[5][5][5]          # 5×5×5 cube

Four-Dimensional Arrays (4D) with Geometric Validation:
triangle_4d = 4D-TRIANGLE[x][y][z][3]   # 3 faces
square_4d = 4D-SQUARE[x][y][z][4]       # 4 faces
pentagon_4d = 4D-PENTAGON[x][y][z][5]   # 5 faces
hexagon_4d = 4D-HEXAGON[x][y][z][6]     # 6 faces (Rubik's cube)
heptagon_4d = 4D-HEPTAGON[x][y][z][7]   # 7 faces
octagon_4d = 4D-OCTAGON[x][y][z][8]     # 8 faces
nonagon_4d = 4D-NONAGON[x][y][z][9]     # 9 faces
decagon_4d = 4D-DECAGON[x][y][z][10]    # 10 faces

Compile-Time Validation:
rubik = 4D-HEXAGON[3][3][3][6]     # ✅ Valid: 6 faces for hexagon
wrong = 4D-HEXAGON[3][3][3][5]     # ❌ Error: Hexagon requires 6 faces

4. Syntax and Semantics
4.1 Function Definition
Correct Syntax:
(: calculate_area(length, width)$
    area = length * width
    perimeter = 2 * (length + width)
    see("Perimeter:", perimeter)
    {[RETURN: (area)]}

Key Features:
(: prefix for function definitionFunction name and parameters$ terminates the headerPython-like indentation for bodyExplicit return: {[RETURN: (value)]}
More Examples:
# Simple factorial
(: factorial(n)$
    if@ n <= 1$
        {[RETURN: (1)]}
    else$
        {[RETURN: (n * factorial(n - 1))]}

# Multiple local variables
(: process_data(input)$
    normalized = input / 100.0
    squared = normalized * normalized
    result = squared + 2.5
    {[RETURN: (result)]}

# Array processing
(: sum_array(arr, index, accumulator)$
    if@ index >= length(arr)$
        {[RETURN: (accumulator)]}
    else$
        new_acc = accumulator + arr[index]
        {[RETURN: (sum_array(arr, index + 1, new_acc))]}

4.2 Conditional Statements
If Statement:
if@ condition$
    statement1
    statement2

If-Else:
if@ temperature > 30$
    see("It's hot!")
    status = "hot"
else$
    see("It's comfortable")
    status = "normal"

If-Elif-Else Chain:
if@ score >= 90$
    grade = 'A'
elfi@ score >= 80$
    grade = 'B'
elfi@ score >= 70$
    grade = 'C'
else$
    grade = 'F'

Switch Statement:
switch@ day$
    case 1$
        day_name = "Monday"
        break
    case 2$
        day_name = "Tuesday"
        break
    default$
        day_name = "Unknown"
        break

4.3 Operators
Arithmetic: +, -, *, /, %, ** (power)
Comparison: ==, !=, <, >, <=, >=
Logical: and, or, not
Bitwise: &, |, ^, <<, >>
Assignment: =
4.4 Input/Output
Output (see):
see("Hello, World!")
see("Value:", x, "Type:", "integer")
see(result)

Input (take):
name = take[string]("Enter your name: ")
age = take[int]("Enter your age: ")
height = take[float]("Enter height: ")
is_student = take[bool]("Are you a student? ")

5. Block Structure System
5.1 Main Block
Every executable reQ program must have a main block:
{[MAIN: 1]}-${
    # Program entry point
    # Variable declarations
    # Function calls
    # Thread invocations
    {[RETURN: (nun)]}
}

5.2 Thread Blocks
Threads define concurrent execution units:
{[THREAD: 1]}-${
    # Thread-specific code
    (: process_chunk(data, start, end, index, accumulator)$
        if@ index >= end$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator + data[index]
            {[RETURN: (process_chunk(data, start, end, index + 1, new_acc))]}

    result = process_chunk(dataset, 0, 1000, 0, 0)
    {[RETURN: (result)]}
}

{[THREAD: 2]}-${
    # Another independent thread
    (: process_chunk(data, start, end, index, accumulator)$
        if@ index >= end$
            {[RETURN: (accumulator)]}
        else$
            new_acc = accumulator + data[index]
            {[RETURN: (process_chunk(data, start, end, index + 1, new_acc))]}

    result = process_chunk(dataset, 1000, 2000, 1000, 0)
    {[RETURN: (result)]}
}

5.3 Thread Value Collection
Threads return values that can be collected in the main block:
{[MAIN: 1]}-${
    dataset = 1D[2000]
    fill_array_sequential(dataset, 1)

    # Execute threads and collect results
    sum1 = {[THREAD: 1]}    # Blocks until thread 1 completes
    sum2 = {[THREAD: 2]}    # Blocks until thread 2 completes

    total = sum1 + sum2
    see("Total:", total)

    {[RETURN: (nun)]}
}

5.4 Function Blocks (Optional)
Group related functions for organization:
{[FUNCTION: math_operations]}-${
    (: add(a, b)$
        {[RETURN: (a + b)]}

    (: multiply(a, b)$
        {[RETURN: (a * b)]}

    {[RETURN: (nun)]}
}

6. Array Systems and Multi-Dimensional Data
6.1 Standard Array Operations
# Length
size = length(array)

# Append
array = append(array, new_element)

# Remove
array = remove(array, index)

# Insert
array = insert(array, index, element)

# Sort
sorted_array = sort(array)

# Reverse
reversed_array = reverse(array)

# Statistical
maximum = max(array)
minimum = min(array)
total = sum(array)
average = mean(array)

6.2 Matrix Operations (2D)
# Creation
matrix = 2D[3][4]

# Operations
transposed = transpose_2d(matrix)
sum_matrix = add_2d(matrix1, matrix2)
product = multiply_2d(matrix1, matrix2)
scaled = scale_2d(matrix, factor)

# Statistics
matrix_mean = mean_2d(matrix)
row_sums = sum_rows(matrix)
col_sums = sum_cols(matrix)

6.3 4D Array Operations
Basic Operations:
# Creation
cube = 4D-HEXAGON[5][5][5][6]
fill_4d(cube, 0.0)

# Arithmetic
sum_cube = add_4d(cube1, cube2)
scaled = scale_4d(cube, 2.5)

# Transformations
rotated = rotate_4d(cube, "x", 90)
translated = translate_4d(cube, [1, 2, 3, 0])

Rubik's Cube Operations:
# Initialize
rubik = 4D-HEXAGON[3][3][3][6]
initialize_solved_cube(rubik)

# Moves
move_cube(rubik, "R")    # Right turn
move_cube(rubik, "U")    # Up turn
move_cube(rubik, "F")    # Front turn

# Sequences
moves = ["R", "U", "R'", "U'"]
move_sequence(rubik, moves)

# Solve
solution = solve_cube(rubik)
is_complete = is_solved(rubik)

7. Recursion-Only Programming Model
7.1 Why No Loops?
Traditional loops introduce:
Mutable loop countersComplex break/continue logicDifficult-to-analyze data dependenciesLimited parallelization opportunities
Recursion provides:
Clear base and recursive casesExplicit data flowEasier compiler optimizationNatural tail-call optimization
7.2 Tail Recursion
Tail recursion is when the recursive call is the last operation:
# Tail-recursive factorial
(: factorial_tail(n, accumulator)$
    if@ n <= 1$
        {[RETURN: (accumulator)]}
    else$
        {[RETURN: (factorial_tail(n - 1, n * accumulator))]}

The compiler converts this to efficient assembly with constant memory usage!
7.3 Recursive Array Processing
# Sum array recursively
(: sum_array(arr, index, accumulator)$
    if@ index >= length(arr)$
        {[RETURN: (accumulator)]}
    else$
        {[RETURN: (sum_array(arr, index + 1, accumulator + arr[index]))]}

# Usage
total = sum_array(numbers, 0, 0)

7.4 Multiple Recursion with Memoization
{[MAIN: 1]}-${
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

    result = fibonacci_memo(50, memo)
    see("Fibonacci(50):", result)
    {[RETURN: (nun)]}
}

8. Hyper-Multithreading Architecture
8.1 Thread System Design
reQ supports 100+ concurrent threads through:
Work-Stealing Scheduler: Idle threads steal work from busy threadsAutomatic Load Balancing: Runtime distributes threads across CPU coresZero-Copy Return Values: Efficient data transfer between threadsDeadlock Detection: Runtime monitors for circular dependencies
8.2 Parallel Array Processing Example
{[THREAD: 1]}-${
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
    dataset = 1D[1000]
    fill_random(dataset, 0.0, 100.0)

    # Parallel execution
    sum1 = {[THREAD: 1]}
    sum2 = {[THREAD: 2]}

    total = sum1 + sum2
    see("Sum of squares:", total)
    {[RETURN: (nun)]}
}

8.3 Advanced Threading Patterns
Producer-Consumer:
{[THREAD: 1]}-${
    # Producer
    (: generate_data(size, index, result)$
        if@ index >= size$
            {[RETURN: (result)]}
        else$
            new_result = append(result, index * 2)
            {[RETURN: (generate_data(size, index + 1, new_result))]}

    data = generate_data(1000, 0, [])
    {[RETURN: (data)]}
}

{[THREAD: 2]}-${
    # Consumer
    input = {[THREAD: 1]}

    (: process_data(arr, index, accumulator)$
        if@ index >= length(arr)$
            {[RETURN: (accumulator)]}
        else$
            {[RETURN: (process_data(arr, index + 1, accumulator + arr[index]))]}

    processed = process_data(input, 0, 0)
    {[RETURN: (processed)]}
}

{[MAIN: 1]}-${
    final_result = {[THREAD: 2]}
    see("Processed result:", final_result)
    {[RETURN: (nun)]}
}

9. Multi-Language Embedding System (FRIEND)
9.1 FRIEND System Overview
The FRIEND (Foreign Runtime Integration and Embedded Native Development) system is reQ's revolutionary feature that allows seamless embedding of code from other languages:
C++: For raw performance, system programming, and existing librariesPython: For machine learning, data science, and rapid prototypingJava: For enterprise APIs, Android development, and mature ecosystems
Key Benefits:
Zero-friction integrationAutomatic data marshaling between languagesReturn values flow naturally into reQ codeCompile-time type checking across language boundaries
9.2 FRIEND Syntax Structure
result = {FRIEND["language"]}-${
    # Foreign language code
    # Can use language-specific libraries
    # Must end with return statement
    {[RETURN: (value)]}
}

9.3 Python Embedding
Machine Learning Example:
{[MAIN: 1]}-${
    # Use Python for ML
    py_result = {FRIEND["py"]}-${
        import tensorflow as tf
        import numpy as np

        # Load pre-trained model
        model = tf.keras.models.load_model('model.h5')

        # Make prediction
        input_data = np.array([[1, 2, 3, 4, 5]])
        prediction = model.predict(input_data)

        {[RETURN: (prediction[0][0])]}
    }

    see("Python ML prediction:", py_result)

    # Use prediction in reQ
    if@ py_result > 0.5$
        see("Positive classification")
    else$
        see("Negative classification")

    {[RETURN: (nun)]}
}

Data Processing Example:
{[MAIN: 1]}-${
    # Generate data in reQ
    data = [1.5, 2.7, 3.2, 4.8, 5.1]

    # Process with Python data science libraries
    normalized = {FRIEND["py"]}-${
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Access reQ data (automatic marshaling)
        data_array = np.array(data_from_req).reshape(-1, 1)

        # Apply sklearn transformation
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data_array)

        {[RETURN: (normalized.flatten().tolist())]}
    }

    see("Normalized data:", normalized)
    {[RETURN: (nun)]}
}

9.4 C++ Embedding
Performance-Critical Code:
{[MAIN: 1]}-${
    # Use C++ for performance
    cpp_result = {FRIEND["cpp"]}-${
        #include <vector>
        #include <algorithm>
        #include <numeric>

        // Create large dataset
        std::vector<int> data(1000000);
        std::iota(data.begin(), data.end(), 1);

        // Perform heavy computation
        std::sort(data.begin(), data.end(), std::greater<int>());

        long long sum = 0;
        for(int val : data) {
            sum += val * val;
        }

        {[RETURN: (sum)]}
    }

    see("C++ computed sum:", cpp_result)
    {[RETURN: (nun)]}
}

Graphics/Math Library:
{[MAIN: 1]}-${
    vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]

    area = {FRIEND["cpp"]}-${
        #include <vector>
        #include <cmath>

        struct Point {
            double x, y;
        };

        std::vector<Point> points = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.5, 1.0}
        };

        // Shoelace formula for polygon area
        double area = 0.0;
        int n = points.size();
        for(int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            area += points[i].x * points[j].y;
            area -= points[j].x * points[i].y;
        }
        area = std::abs(area) / 2.0;

        {[RETURN: (area)]}
    }

    see("Triangle area:", area)
    {[RETURN: (nun)]}
}

9.5 Java Embedding
Enterprise Integration:
{[MAIN: 1]}-${
    # Use Java for enterprise APIs
    java_result = {FRIEND["java"]}-${
        import java.util.*;
        import java.util.stream.*;

        // Java collections
        List<Integer> numbers = Arrays.asList(10, 20, 30, 40, 50);

        // Stream processing
        int sum = numbers.stream()
                        .filter(n -> n > 15)
                        .mapToInt(Integer::intValue)
                        .sum();

        {[RETURN: (sum)]}
    }

    see("Java computed sum:", java_result)
    {[RETURN: (nun)]}
}

Android/Mobile Development:
{[MAIN: 1]}-${
    sensor_data = {FRIEND["java"]}-${
        import android.hardware.Sensor;
        import android.hardware.SensorManager;

        // Access Android sensors
        SensorManager sensorManager = getSensorManager();
        Sensor accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        float[] values = getLatestSensorValues(accelerometer);

        {[RETURN: (values[0])]}  # Return X-axis acceleration
    }

    see("Accelerometer X:", sensor_data)
    {[RETURN: (nun)]}
}

9.6 Hybrid Computing Pipeline
Complete Multi-Language Workflow:
{[MAIN: 1]}-${
    # Step 1: Generate data in reQ
    raw_data = 1D[1000]
    fill_random(raw_data, 0.0, 100.0)

    # Step 2: Process with Python ML
    ml_features = {FRIEND["py"]}-${
        import numpy as np
        from sklearn.decomposition import PCA

        # Access reQ data
        data = np.array(data_from_req).reshape(-1, 1)

        # Apply PCA
        pca = PCA(n_components=1)
        features = pca.fit_transform(data)

        # Normalize
        normalized = (features - features.mean()) / features.std()

        {[RETURN: (normalized.flatten().tolist())]}
    }

    # Step 3: Heavy computation in C++
    processed = {FRIEND["cpp"]}-${
        #include <vector>
        #include <cmath>

        std::vector<double> input = ml_features_from_req;
        std::vector<double> result;

        // Complex mathematical transformation
        for(double val : input) {
            double transformed = std::pow(val, 3) + 2 * std::pow(val, 2) + val + 1;
            result.push_back(transformed);
        }

        {[RETURN: (result)]}
    }

    # Step 4: Aggregate in Java
    summary = {FRIEND["java"]}-${
        import java.util.*;
        import java.util.stream.*;

        List<Double> data = Arrays.asList(processed_from_req);

        DoubleSummaryStatistics stats = data.stream()
            .mapToDouble(Double::doubleValue)
            .summaryStatistics();

        {[RETURN: (stats.getAverage())]}
    }

    # Step 5: Final aggregation in reQ
    (: sum_list(lst, idx, acc)$
        if@ idx >= length(lst)$
            {[RETURN: (acc)]}
        else$
            {[RETURN: (sum_list(lst, idx + 1, acc + lst[idx]))]}

    total = sum_list(processed, 0, 0.0)
    average = total / length(processed)

    see("Java summary:", summary)
    see("reQ average:", average)
    see("Pipeline complete!")

    {[RETURN: (nun)]}
}

9.7 Data Marshaling
Automatic Type Conversion:
reQ Type Python C++ Java int int int int float float double double string str std::string String bool bool bool boolean 1D array list std::vector List 2D array np.array std::vector<std::vector> List<List>
Data Flow:
reQ variables are automatically available in FRIEND blocksForeign code accesses them with _from_req suffixReturn values are automatically converted back to reQ typesType checking occurs at compile time
9.8 FRIEND Best Practices
When to Use Each Language:
C++: Low-level operations, graphics, system calls, performance-critical codePython: ML/AI, data science, rapid prototyping, scientific computingJava: Enterprise integration, Android development, established APIs
Performance Considerations:
Minimize data transfer between languagesBatch operations when possibleUse C++ for inner loopsUse Python for high-level orchestration
Error Handling:
{[MAIN: 1]}-${
    try$
        result = {FRIEND["py"]}-${
            # Risky operation
            import some_module
            {[RETURN: (some_module.compute())]}
        }
        see("Success:", result)
    exept-> error$
        see("Python error:", error)
        result = 0

    {[RETURN: (nun)]}
}

10. Memory Management
10.1 Automatic Memory Management
reQ provides garbage collection with:
Reference Counting: Immediate cleanup when references drop to zeroGenerational GC: Young, middle, and old generation for efficiencyMemory Pools: Fast allocation for common sizesZero-Copy Optimization: Minimize data copying
10.2 Memory Allocation
Stack Allocation (automatic):
Primitive typesSmall structuresFunction parameters
Heap Allocation (automatic):
Large arraysStrings4D arraysFRIEND block return values
10.3 Thread Memory
Each thread has:
Thread-local storage: Variables local to the threadShared memory access: Automatic synchronizationReturn value optimization: Zero-copy when possible
10.4 Cross-Language Memory
FRIEND blocks share memory through:
Automatic serialization/deserializationCopy-on-write for large data structuresReference passing for read-only data
11. Exception Handling
11.1 Try-Except Syntax
try$
    result = risky_operation()
    see("Success:", result)
exept-> error$
    see("Error:", error)
    result = default_value()

11.2 Common Exception Types
DivisionByZero: Division by zero attemptedIndexOutOfBounds: Array access beyond boundsTypeMismatch: Incompatible type operationNullReference: Operation on null valueThreadTimeout: Thread execution exceeded limitInvalidArrayShape: 4D array shape mismatchForeignLanguageError: Error in FRIEND block executionCompilationError: Foreign code compilation failed
11.3 Multiple Exception Handling
try$
    result = complex_operation()
exept-> DivisionByZero$
    see("Cannot divide by zero")
    result = 0
exept-> IndexOutOfBounds$
    see("Array access error")
    result = -1
exept-> ForeignLanguageError$
    see("FRIEND block failed")
    result = -999
exept-> error$
    see("Unknown error:", error)
    result = -999

11.4 Exception Handling with FRIEND Blocks
{[MAIN: 1]}-${
    try$
        py_result = {FRIEND["py"]}-${
            import tensorflow as tf
            # This might fail if model doesn't exist
            model = tf.keras.models.load_model('model.h5')
            {[RETURN: (1)]}
        }
        see("Model loaded successfully")
    exept-> ForeignLanguageError$
        see("Failed to load Python model")
        py_result = 0

    {[RETURN: (nun)]}
}

12. Standard Library
12.1 Mathematical Functions
abs(x)              # Absolute value
sqrt(x)             # Square root
pow(x, y)           # Power
sin(x), cos(x), tan(x)  # Trigonometry
asin(x), acos(x), atan(x)  # Inverse trig
log(x), log10(x)    # Logarithms
exp(x)              # Exponential
floor(x), ceil(x)   # Rounding
round(x, digits)    # Round to digits

12.2 Array Functions
length(arr)         # Length
append(arr, item)   # Add element
remove(arr, idx)    # Remove element
insert(arr, idx, el)  # Insert element
sort(arr)           # Sort ascending
sort_desc(arr)      # Sort descending
reverse(arr)        # Reverse
max(arr), min(arr)  # Extrema
sum(arr), mean(arr) # Statistics
median(arr)         # Median
variance(arr)       # Variance
std_dev(arr)        # Standard deviation

12.3 String Functions
strlen(s)           # Length
strcat(s1, s2)      # Concatenate
substr(s, start, len)  # Substring
str_upper(s)        # Uppercase
str_lower(s)        # Lowercase
split(s, delim)     # Split
strip(s)            # Remove whitespace
replace(s, old, new)  # Replace substring
find(s, substring)  # Find index
startswith(s, prefix)  # Check prefix
endswith(s, suffix)  # Check suffix

12.4 2D/3D Array Functions
# 2D Operations
transpose_2d(matrix)
add_2d(m1, m2)
multiply_2d(m1, m2)
scale_2d(matrix, factor)
determinant_2d(matrix)
inverse_2d(matrix)
mean_2d(matrix)
sum_rows(matrix)
sum_cols(matrix)

# 3D Operations
add_3d(v1, v2)
scale_3d(volume, factor)
mean_3d(volume)
max_3d(volume)
rotate_3d(volume, axis, angle)

12.5 4D Array Library
# Creation & Initialization
create_4d(x, y, z, shape, faces)
fill_4d(cube, value)
copy_4d(cube)

# Arithmetic Operations
add_4d(a, b)
subtract_4d(a, b)
multiply_4d(a, b)
scale_4d(cube, factor)

# Transformations
rotate_4d(cube, axis, angle)
translate_4d(cube, vector)
transpose_4d(cube)

# Statistics
mean_4d(cube)
variance_4d(cube)
sum_4d(cube)
max_4d(cube)
min_4d(cube)

# Rubik's Cube Specific
initialize_solved_cube(cube)
scramble_cube(cube)
move_cube(cube, move)  # "R", "L", "U", "D", "F", "B"
move_sequence(cube, moves)
solve_cube(cube)
is_solved(cube)
get_face(cube, face_index)
set_face(cube, face_index, data)

12.6 Utility Functions
# Type Conversion
int_to_float(x)
float_to_int(x)
to_string(x)
parse_int(s)
parse_float(s)

# Random Numbers
random()            # 0.0 to 1.0
random_int(min, max)  # Integer range
random_float(min, max)  # Float range
set_seed(seed)      # Set random seed

# Time
current_time()      # Current timestamp
sleep(seconds)      # Sleep for duration

13. Compiler Architecture
13.1 Compilation Pipeline
Lexical Analysis: Source → TokensSyntax Analysis: Tokens → ASTSemantic Analysis: Type checking, FRIEND validationForeign Code Compilation: Compile embedded C++/Python/JavaOptimization: Tail-call, dead code eliminationCode Generation: Assembly generationForeign Code Linking: Link compiled FRIEND blocksAssembly & Linking: Final executable
13.2 Compiler Commands
# Basic compilation
beetroot compile program.req

# With optimization levels
beetroot compile program.req -O1  # Basic optimization
beetroot compile program.req -O2  # Moderate optimization
beetroot compile program.req -O3  # Aggressive optimization

# Specify output
beetroot compile program.req -o my_executable

# Debug mode (preserve symbols)
beetroot compile program.req -g

# Assembly output
beetroot compile program.req -S

# Verbose compilation
beetroot compile program.req -v

# Check FRIEND dependencies
beetroot check-deps program.req

# Compile with specific language support
beetroot compile program.req --enable-python --enable-cpp --enable-java

13.3 Optimizations
Core Optimizations:
Tail Call Optimization: Recursion → loopsConstant Folding: Compile-time evaluationDead Code Elimination: Remove unused codeRegister Allocation: Efficient register usageInline Expansion: Small function inlining
Thread Optimizations:
Thread Fusion: Combine compatible threadsWork Stealing: Dynamic load balancingThread Pool Management: Reuse threads
FRIEND Optimizations:
Foreign Code Caching: Cache compiled foreign codeData Marshaling Optimization: Minimize copiesLazy Compilation: Compile FRIEND blocks on demand
13.4 FRIEND Compilation Process
Extract FRIEND Blocks: Identify all embedded codeLanguage Detection: Determine language from {FRIEND["lang"]}Separate Compilation: Compile each FRIEND block independentlyInterface Generation: Create reQ ↔ Foreign language interfacesLinking: Link compiled foreign code with reQ executableValidation: Verify type compatibility across boundaries
14. Development Tools
14.1 Debugger (reqdb)
reqdb executable

# Commands:
break main          # Set breakpoint at main
break function_name # Set breakpoint at function
run                 # Start execution
step                # Step into function
next                # Step over function
continue            # Continue execution
print variable      # Print variable value
backtrace           # Show call stack
thread list         # List all threads
thread switch N     # Switch to thread N
watch variable      # Watch variable changes

14.2 Profiler (reqprof)
reqprof executable

# Output includes:
# - Function timing analysis
# - Thread performance metrics
# - Memory usage statistics
# - FRIEND block overhead
# - Call graph visualization
# - Hotspot identification

# Advanced profiling
reqprof executable --thread-analysis
reqprof executable --memory-trace
reqprof executable --foreign-overhead

14.3 Package Manager (reqpkg)
# Install packages
reqpkg install math-extended
reqpkg install graphics-3d
reqpkg install ml-toolkit

# Update packages
reqpkg update
reqpkg update math-extended

# Search packages
reqpkg search machine-learning

# List installed
reqpkg list

# Remove package
reqpkg remove graphics-3d

# Package information
reqpkg info ml-toolkit

14.4 FRIEND Manager (reqfriend)
# Check language support
reqfriend check python
reqfriend check cpp
reqfriend check java

# Install language runtimes
reqfriend install python-runtime
reqfriend install cpp-compiler
reqfriend install java-jdk

# Test FRIEND compilation
reqfriend test program.req

# Show foreign dependencies
reqfriend deps program.req

# Update language bindings
reqfriend update-bindings

14.5 Interactive REPL (reqrepl)
reqrepl

# Interactive mode:
>>> x = 10
>>> y = 20
>>> see(x + y)
30

>>> (: factorial(n)$
...     if@ n <= 1$
...         {[RETURN: (1)]}
...     else$
...         {[RETURN: (n * factorial(n - 1))]}
...
>>> see(factorial(5))
120

>>> py_result = {FRIEND["py"]}-${
...     import math
...     {[RETURN: (math.pi)]}
... }
>>> see(py_result)
3.141592653589793

15. Complete Language Reference
15.1 Keywords
# Type Keywords
int float char string bool nun

# Control Flow
if elfi else switch case default break

# Exception Handling
try exept

# Block Structure
MAIN THREAD FUNCTION RETURN FRIEND

# Operators/Logic
and or not
true false

# Array Dimensions
1D 2D 3D 4D-TRIANGLE 4D-SQUARE 4D-PENTAGON 4D-HEXAGON
4D-HEPTAGON 4D-OCTAGON 4D-NONAGON 4D-DECAGON

15.2 Operators
Arithmetic: + - * / % **
Comparison: == != < > <= >=
Logical: and or not
Bitwise: & | ^ << >>
Assignment: =
15.3 Special Symbols
Symbol Purpose Example (: Function definition start `(: func(x)$` $ Statement terminator `if@ x > 0$` `@` Condition marker `if@`, `switch@` `{[...]}-${...}` Block structure `{[MAIN: 1]}-${...}` `{[RETURN: (...)]}` Return statement `{[RETURN: (value)]}` `{FRIEND["lang"]}-${...}` Foreign code embedding `{FRIEND["py"]}-${...}`

15.4 Built-in Functions Quick Reference
# I/O
see(...)            # Output
take[type](prompt)  # Input

# Math
abs, sqrt, pow, sin, cos, tan, log, exp, floor, ceil, round

# Arrays
length, append, remove, insert, sort, reverse, max, min, sum, mean

# Strings
strlen, strcat, substr, str_upper, str_lower, split, strip

# Type Conversion
int_to_float, float_to_int, to_string, parse_int, parse_float

# Random
random, random_int, random_float, set_seed

# Time
current_time, sleep

# 4D Arrays
create_4d, fill_4d, add_4d, rotate_4d, solve_cube, is_solved

15.5 Complete Syntax Examples
Full Program Template:
{[MAIN: 1]}-${
    # Variable declarations
    x = 10
    y = 20

    # Function definition
    (: calculate(a, b)$
        result = a * b + a / b
        {[RETURN: (result)]}

    # Call function
    value = calculate(x, y)
    see("Result:", value)

    # Use FRIEND blocks
    py_data = {FRIEND["py"]}-${
        import numpy as np
        data = np.random.randn(100)
        {[RETURN: (data.mean())]}
    }

    cpp_result = {FRIEND["cpp"]}-${
        #include <cmath>
        double result = std::pow(2.0, 10.0);
        {[RETURN: (result)]}
    }

    # Use results
    see("Python mean:", py_data)
    see("C++ power:", cpp_result)

    # Conditional logic
    if@ py_data > 0$
        see("Positive mean")
    else$
        see("Negative mean")

    # Exception handling
    try$
        risky = calculate(x, 0)
    exept-> DivisionByZero$
        see("Division by zero!")
        risky = 0

    {[RETURN: (nun)]}
}

Multithreaded with FRIEND:
{[THREAD: 1]}-${
    # Python ML in thread
    prediction = {FRIEND["py"]}-${
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5')
        result = model.predict([[1, 2, 3]])
        {[RETURN: (result[0][0])]}
    }
    {[RETURN: (prediction)]}
}

{[THREAD: 2]}-${
    # C++ processing in thread
    processed = {FRIEND["cpp"]}-${
        #include <vector>
        #include <algorithm>
        std::vector<int> data(10000);
        std::sort(data.begin(), data.end());
        {[RETURN: (data.size())]}
    }
    {[RETURN: (processed)]}
}

{[MAIN: 1]}-${
    # Collect results
    ml_result = {[THREAD: 1]}
    cpp_result = {[THREAD: 2]}

    see("ML prediction:", ml_result)
    see("C++ processed:", cpp_result)

    {[RETURN: (nun)]}
}

Conclusion
reQ Programming Language v2.0 represents a revolutionary approach to high-performance computing by combining:
Assembly-Level Performance: Direct compilation to optimized machine codePure Recursion Model: Elegant, optimizable computation patternsHyper-Multithreading: Native support for 100+ concurrent threadsGeometric Arrays: Unique 4D arrays with compile-time shape validationMulti-Language Integration: Seamless embedding of C++, Python, and Java through the FRIEND system
The FRIEND system is particularly groundbreaking, allowing developers to:
Leverage Python's ML ecosystem for AI/data scienceUse C++ for performance-critical sectionsIntegrate Java for enterprise/Android applicationsMaintain type safety across language boundariesAchieve optimal performance for each task
Key Syntax Reminders
✅ Function definition: `(: function_name(params)$`
✅ Dollar sign terminator: Required after function headers and conditionals
✅ Python-like indentation: For function bodies
✅ Explicit returns: `{[RETURN: (value)]}`
✅ FRIEND blocks: `{FRIEND["lang"]}-${...}`

When to Use reQ
Perfect for:
- Scientific computing with ML integration
- High-performance graphics with multiple language ecosystems
- Parallel data processing pipelines
- Hybrid applications requiring Python, C++, and Java
- Mathematical simulations with 4D geometric data

Example Use Cases:
- ML model training (Python) + inference optimization (C++)
- Game engines with scripting (Python) + core engine (C++)
- Data pipelines with processing (Java) + analytics (Python)
- Scientific simulations with visualization (C++) + analysis (Python)

reQ empowers developers to choose the right tool for each task while maintaining performance, type safety, and elegant code structure.
