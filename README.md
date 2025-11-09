# reQ Programming Language

Welcome to reQ, a revolutionary programming language designed for high-performance computing, mathematical correctness, and seamless multi-language integration.

reQ (recursive Queue) is a compiled, statically-typed language that combines the performance of C, the developer-friendly syntax of Python, and a unique, recursion-only programming model. Its standout feature is the **FRIEND (Foreign Runtime Integration and Embedded Native Development)** system, which allows you to embed C++, Python, and Java code directly into your reQ programs.

## Core Philosophy

- **Performance First**: reQ compiles directly to optimized assembly with the `beetroot` compiler, targeting performance within 5% of hand-optimized C.
- **Pure Recursion**: All repetitive operations are handled through recursion, enabling powerful compiler optimizations like automatic tail-call optimization.
- **Explicit Parallelism**: Hyper-multithreading is a first-class citizen, with support for 100+ concurrent threads managed by a work-stealing scheduler.
- **Polyglot Integration**: The FRIEND system lets you leverage the C++, Python, and Java ecosystems without friction.

## At a Glance: "Hello, World!"

```req
{[MAIN: 1]}-${
    see("Hello, World!")
    {[RETURN: (nun)]}
}
```

## Key Features

### 1. Simple, Indentation-Based Syntax

reQ's syntax is inspired by Python, using indentation to define blocks. Functions are defined with `(:`, and headers are terminated with `$`.

```req
(: factorial(n, accumulator)$
    if@ n <= 1$
        {[RETURN: (accumulator)]}
    else$
        {[RETURN: (factorial_tail(n - 1, n * accumulator))]}
```

### 2. The FRIEND System: Multi-Language Embedding

Seamlessly embed Python, C++, and Java. reQ handles data marshaling automatically, allowing you to mix and match languages to use the best tool for the job.

**Example: Python for ML, C++ for computation**
```req
{[MAIN: 1]}-${
    # Step 1: Get ML prediction from a Python block
    prediction = {FRIEND["py"]}-${
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5')
        result = model.predict([[1, 2, 3]])
        {[RETURN: (result[0][0])]}
    }

    # Step 2: Use the prediction in a C++ block for heavy computation
    processed_result = {FRIEND["cpp"]}-${
        #include <cmath>
        double input = prediction_from_req; // Data from reQ
        double transformed = std::pow(input, 3);
        {[RETURN: (transformed)]}
    }

    see("Final Result:", processed_result)
    {[RETURN: (nun)]}
}
```

### 3. Hyper-Multithreading

reQ is built for concurrency. Define `THREAD` blocks and collect their results effortlessly. The runtime handles scheduling and load balancing.

```req
{[THREAD: 1]}-${
    {[RETURN: (sum_array(data, 0, 500, 0))]}
}

{[THREAD: 2]}-${
    {[RETURN: (sum_array(data, 500, 1000, 0))]}
}

{[MAIN: 1]}-${
    sum1 = {[THREAD: 1]} # Blocks until complete
    sum2 = {[THREAD: 2]}
    total = sum1 + sum2
    see("Total:", total)
    {[RETURN: (nun)]}
}
```

### 4. Advanced Array System with Geometric Validation

reQ has native support for multi-dimensional arrays, including unique 4D arrays with compile-time geometric validation. This is perfect for simulations, graphics, and advanced mathematical modeling.

```req
# A valid 4D Rubik's Cube (hexagon with 6 faces)
rubik = 4D-HEXAGON[3][3][3][6]     # ✅ Compiles

# An invalid shape
wrong = 4D-HEXAGON[3][3][3][5]     # ❌ Compile-time error
```

## Getting Started

To compile a reQ program, use the `beetroot` compiler:

```bash
beetroot compile your_program.req -o executable
```

## Development Tools

reQ comes with a suite of tools to enhance productivity:
- **`reqdb`**: A powerful debugger for stepping through code, inspecting variables, and managing threads.
- **`reqprof`**: A profiler to identify performance bottlenecks in both reQ and FRIEND blocks.
- **`reqpkg`**: A package manager for installing and managing reQ libraries.
- **`reqfriend`**: A tool to manage foreign language runtimes (Python, C++, Java) and dependencies.
- **`reqrepl`**: An interactive REPL for quick prototyping and testing.

Dive into the `docs/` directory to explore the complete language specification.
