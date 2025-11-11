// Color configuration for syntax highlighting
const COLORS = {
    keyword: '#2fff00ff',      // Keywords - bright green
    userDefined: '#b70733ff',  // User-defined - dark red
    special: '#056bf9ff',      // Special characters - bright blue
    string: '#FF6B9D',       // Strings - pink
    comment: '#888888'       // Comments - gray
};

// Code examples for hero section
const codeExamples = [
    {
        title: "Hello World",
        code: `{[MAIN: 1]}-\${
    see("Hello, reQ World!")
    {[RETURN: (nun)]}
}`
    },
    {
        title: "Recursive Factorial",
        code: `(: factorial(n)$
    if@ n <= 1$
        {[RETURN: (1)]}
    else$
        {[RETURN: (n * factorial(n - 1))]}`
    },
    {
        title: "FRIEND Python Integration",
        code: `py_result = {FRIEND["py"]}-\${
    import numpy as np
    data = np.array([1,2,3])
    {[RETURN: (data.mean())]}
}`
    },
    {
        title: "Multi-Threading",
        code: `{[THREAD: 1]}-\${
    result = process_data(dataset)
    {[RETURN: (result)]}
}

sum1 = {[THREAD: 1]}`
    }
];

// Documentation sections
const sections = {
    overview: {
        title: 'Language Overview',
        icon: 'book',
        content: `reQ (recursive Queue) - Revolutionary High-Performance Programming Language

Key Innovations:
• Assembly compilation for maximum speed
• Recursion-only paradigm (no loops)
• Hyper-multithreading (100+ concurrent threads)
• Native 4D arrays with geometric validation
• Automatic memory management
• Multi-language embedding support

Created by: CVAKI
Compiler: beetroot
Performance: Within 5% of hand-optimized C code`
    },
    syntax: {
        title: 'Core Syntax',
        icon: 'code',
        content: `Block Structure:
{[BLOCK_TYPE: ID]}-\${
    # Code body
    {[RETURN: (value)]}
}

Function Definition:
(: function_name(param1, param2)$
    # Function body
    {[RETURN: (expression)]}

Control Flow:
if@ condition$ statements
elfi@ condition$ statements
else$ statements

Switch Statement:
switch@ variable$
    case value$ statements
    default$ statements`
    },
    threads: {
        title: 'Hyper-Threading',
        icon: 'cpu',
        content: `Thread Declaration:
{[THREAD: 1]}-\${
    result = computation()
    {[RETURN: (result)]}
}

Thread Execution:
{[MAIN: 1]}-\${
    value = {[THREAD: 1]}
    {[RETURN: (nun)]}
}

Features:
• 100+ concurrent threads
• Automatic scheduling
• Work-stealing algorithm
• Return value collection
• Zero-copy optimization
• Deadlock detection`
    },
    arrays: {
        title: '4D Arrays',
        icon: 'database',
        content: `Geometric 4D Arrays:
4D-TRIANGLE[x][y][z][3]   # 3 faces
4D-SQUARE[x][y][z][4]     # 4 faces
4D-PENTAGON[x][y][z][5]   # 5 faces
4D-HEXAGON[x][y][z][6]    # 6 faces (Rubik's cube)
4D-HEPTAGON[x][y][z][7]   # 7 faces
4D-OCTAGON[x][y][z][8]    # 8 faces
4D-NONAGON[x][y][z][9]    # 9 faces
4D-DECAGON[x][y][z][10]   # 10 faces

Compile-time validation ensures:
• Correct face counts
• Type safety
• Optimized memory layout
• Geometric consistency`
    },
    recursion: {
        title: 'Recursion Patterns',
        icon: 'git-branch',
        content: `Tail Recursion (Optimized):
(: factorial_tail(n, acc)$
    if@ n <= 1$ {[RETURN: (acc)]}
    else$ {[RETURN: (factorial_tail(n - 1, n * acc))]}

Multiple Recursion:
(: fibonacci(n)$
    if@ n <= 1$ {[RETURN: (n)]}
    else$ {[RETURN: (fibonacci(n-1) + fibonacci(n-2))]}

Array Processing:
(: sum_array(arr, index, accumulator)$
    if@ index >= length(arr)$
        {[RETURN: (accumulator)]}
    else$
        {[RETURN: (sum_array(arr, index + 1, accumulator + arr[index]))]}

Compiler automatically:
• Converts tail calls to loops
• Optimizes memory usage
• Identifies parallelization opportunities`
    },
    embedding: {
        title: 'Language Embedding',
        icon: 'globe',
        content: `FRIEND System: Multi-Language Support

Embed C++:
{FRIEND["cpp"]}-\${
    #include <vector>
    std::vector<int> vec = {1, 2, 3};
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    {[RETURN: (sum)]}
}

Embed Python:
{FRIEND["py"]}-\${
    import numpy as np
    result = np.array([1, 2, 3]).mean()
    {[RETURN: (result)]}
}

Embed Java:
{FRIEND["java"]}-\${
    ArrayList<Integer> list = new ArrayList<>();
    list.add(42);
    {[RETURN: (list.get(0))]}
}

Features:
• Zero-friction integration
• Automatic data marshaling
• Compile-time type checking
• Native library access`
    }
};

// Code examples for documentation
const examples = {
    basic: {
        title: 'Hello World',
        code: `{[MAIN: 1]}-\${
    see("Hello, reQ!")
    name = take[string]("Enter name: ")
    see("Welcome,", name)
    {[RETURN: (nun)]}
}`
    },
    threading: {
        title: 'Parallel Computing',
        code: `{[THREAD: 1]}-\${
    (: sum_range(start, end, acc)$
        if@ start > end$ {[RETURN: (acc)]}
        else$ {[RETURN: (sum_range(start + 1, end, acc + start))]}
    
    result = sum_range(1, 1000, 0)
    {[RETURN: (result)]}
}

{[MAIN: 1]}-\${
    sum = {[THREAD: 1]}
    see("Sum:", sum)
    {[RETURN: (nun)]}
}`
    },
    array4d: {
        title: "Rubik's Cube",
        code: `{[MAIN: 1]}-\${
    # Create 3x3x3 Rubik's cube
    rubik = 4D-HEXAGON[3][3][3][6]
    initialize_solved_cube(rubik)
    
    # Scramble
    moves = ["R", "U", "R'", "U'"]
    move_sequence(rubik, moves)
    
    # Solve
    solution = solve_cube(rubik)
    see("Solution:", solution)
    {[RETURN: (nun)]}
}`
    },
    embedding: {
        title: 'Language Embedding',
        code: `{[MAIN: 1]}-\${
    # Use Python for ML
    py_result = {FRIEND["py"]}-\${
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5')
        prediction = model.predict([[1, 2, 3]])
        {[RETURN: (prediction[0][0])]}
    }
    
    # Use C++ for performance
    cpp_result = {FRIEND["cpp"]}-\${
        #include <algorithm>
        std::vector<int> data(1000000);
        std::sort(data.begin(), data.end());
        {[RETURN: (data.size())]}
    }
    
    see("Python result:", py_result)
    see("C++ result:", cpp_result)
    {[RETURN: (nun)]}
}`
    },
    matrix: {
        title: 'Matrix Operations',
        code: `{[MAIN: 1]}-\${
    # Create matrices
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    
    # Operations
    sum = add_2d(a, b)
    product = multiply_2d(a, b)
    transposed = transpose_2d(a)
    
    see("Product:", product)
    {[RETURN: (nun)]}
}`
    }
};

// Features data
const features = [
    {
        icon: 'zap',
        title: 'Assembly Speed',
        desc: 'Compiles directly to optimized machine code via beetroot compiler'
    },
    {
        icon: 'cpu',
        title: '100+ Threads',
        desc: 'Automatic work-stealing scheduler for maximum parallelism'
    },
    {
        icon: 'database',
        title: '4D Arrays',
        desc: 'Geometric validation at compile time with native support'
    },
    {
        icon: 'git-branch',
        title: 'Pure Recursion',
        desc: 'No loops - automatic tail call optimization to assembly'
    },
    {
        icon: 'file-code',
        title: 'Type Safety',
        desc: 'Strong static typing with intelligent type inference'
    },
    {
        icon: 'globe',
        title: 'Multi-Language',
        desc: 'Embed C++, Python, Java seamlessly with FRIEND system'
    }
];

// SVG icons
const icons = {
    'book': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />',
    'code': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />',
    'cpu': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />',
    'zap': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />',
    'database': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />',
    'git-branch': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />',
    'file-code': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />',
    'globe': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />'
};

// Performance benchmarks
const benchmarks = [
    { name: 'Matrix Multiplication (1024×1024)', value: '125.3ms', percent: 95 },
    { name: 'Fibonacci(40) Recursive', value: '342.5ms', percent: 92 },
    { name: '8-Thread Parallel Processing', value: '7.2x speedup', percent: 90 },
    { name: '4D Array Operations', value: '89.1ms', percent: 93 }
];

// State management
let currentCodeIndex = 0;
let activeSection = 'overview';
let activeExample = 'basic';

// Utility function to get SVG icon
function getIcon(name) {
    return `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">${icons[name]}</svg>`;
}

// Syntax highlighting function - FIXED VERSION
function highlightCode(code) {
    const keywords = ['if@', 'else$', 'elfi@', 'MAIN:', 'THREAD:', 'FUNCTION:', 'RETURN:', 'FRIEND', 'try$', 'exept->', 'switch@', 'case', 'default', 'break', 'import', 'include'];
    const userDefined = ['factorial', 'process_data', 'dataset', 'see', 'take', 'py_result', 'cpp_result', 'data', 'result', 'sum1', 'n', 'acc', 'rubik', 'moves', 'solution', 'name', 'sum', 'product', 'transposed', 'mean', 'array', 'np', 'numpy', 'prediction', 'model', 'vec', 'list'];
    
    // First escape HTML to prevent any issues
    let highlighted = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Highlight comments first (to avoid interfering with other patterns)
    highlighted = highlighted.replace(/#(.*)$/gm, '<span class="comment">#$1</span>');
    
    // Highlight strings (before other replacements to protect string contents)
    highlighted = highlighted.replace(/"([^"]*)"/g, '<span class="string">"$1"</span>');
    highlighted = highlighted.replace(/'([^']*)'/g, "<span class='string'>$1</span>");
    
    // Highlight keywords
    keywords.forEach(keyword => {
        const escapedKeyword = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`\\b(${escapedKeyword})\\b`, 'g');
        highlighted = highlighted.replace(regex, '<span class="keyword">$1</span>');
    });
    
    // Highlight user-defined identifiers (avoid already highlighted text)
    userDefined.forEach(word => {
        const regex = new RegExp(`(?<!<[^>]*)\\b(${word})\\b(?![^<]*>)`, 'g');
        highlighted = highlighted.replace(regex, '<span class="user-defined">$1</span>');
    });
    
    // Highlight special characters individually (avoid already highlighted text)
    highlighted = highlighted.replace(/(?<!<[^>]*)(\{|\}|\[|\]|\(|\)|\$|->|:(?!\/)|@|,)(?![^<]*>)/g, '<span class="special">$1</span>');
    
    return highlighted;
}

// Create floating particles
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.width = Math.random() * 100 + 50 + 'px';
        particle.style.height = particle.style.width;
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDuration = Math.random() * 10 + 5 + 's';
        particle.style.animationDelay = Math.random() * 5 + 's';
        container.appendChild(particle);
    }
}

// Update hero code display
function updateHeroCode() {
    const example = codeExamples[currentCodeIndex];
    const titleEl = document.getElementById('codeTitle');
    const displayEl = document.getElementById('codeDisplay');
    
    if (titleEl) titleEl.textContent = example.title;
    if (displayEl) displayEl.innerHTML = highlightCode(example.code);
    
    currentCodeIndex = (currentCodeIndex + 1) % codeExamples.length;
}

// Render features grid
function renderFeatures() {
    const grid = document.getElementById('featuresGrid');
    if (!grid) return;
    
    grid.innerHTML = features.map(feature => `
        <div class="feature-card">
            <div class="feature-icon">${getIcon(feature.icon)}</div>
            <h3 class="feature-title">${feature.title}</h3>
            <p class="feature-desc">${feature.desc}</p>
        </div>
    `).join('');
}

// Render syntax examples
function renderSyntaxExamples() {
    const example1 = `(: calculate_area(length, width)$
    area = length * width
    {[RETURN: (area)]}`;
    
    const example2 = `{[THREAD: 1]}-\${
    result = process_chunk(data)
    {[RETURN: (result)]}
}`;
    
    const el1 = document.getElementById('syntaxExample1');
    const el2 = document.getElementById('syntaxExample2');
    
    if (el1) el1.innerHTML = highlightCode(example1);
    if (el2) el2.innerHTML = highlightCode(example2);
}

// Render navigation buttons
function renderNavButtons() {
    const nav = document.getElementById('navButtons');
    if (!nav) return;
    
    nav.innerHTML = Object.entries(sections).map(([key, section]) => `
        <button class="nav-button ${activeSection === key ? 'active' : ''}" onclick="setActiveSection('${key}')">
            ${getIcon(section.icon)}
            <span>${section.title}</span>
        </button>
    `).join('');
}

// Render section content
function renderSectionContent() {
    const section = sections[activeSection];
    const contentEl = document.getElementById('docContent');
    if (!contentEl) return;
    
    contentEl.innerHTML = `
        <div class="content-header">
            ${getIcon(section.icon)}
            <h2>${section.title}</h2>
        </div>
        <pre class="content-text">${section.content}</pre>
    `;
}

// Render example tabs
function renderExampleTabs() {
    const tabs = document.getElementById('exampleTabs');
    if (!tabs) return;
    
    tabs.innerHTML = Object.entries(examples).map(([key, example]) => `
        <button class="example-tab ${activeExample === key ? 'active' : ''}" onclick="setActiveExample('${key}')">
            ${example.title}
        </button>
    `).join('');
}

// Render example code
function renderExampleCode() {
    const example = examples[activeExample];
    const codeEl = document.getElementById('exampleCode');
    if (!codeEl) return;
    
    codeEl.innerHTML = highlightCode(example.code);
}

// Render embedding highlight
function renderEmbeddingHighlight() {
    const div = document.getElementById('embeddingHighlight');
    if (!div) return;
    
    if (activeSection === 'embedding') {
        div.innerHTML = `
            <div class="embedding-card">
                <h3 class="embedding-title">
                    ${getIcon('globe')}
                    Multi-Language Embedding
                </h3>
                <p class="embedding-desc">
                    reQ supports seamless embedding of C++, Python, and Java code blocks within your program. 
                    Each embedded block maintains its native syntax and returns values to the main reQ context.
                </p>
                <div class="lang-grid">
                    <div class="lang-card">
                        <code style="color: #fb923c; font-weight: bold;">cpp</code>
                        <p>C++ Standard Library</p>
                    </div>
                    <div class="lang-card">
                        <code style="color: #fbbf24; font-weight: bold;">py</code>
                        <p>Python with NumPy/ML</p>
                    </div>
                    <div class="lang-card">
                        <code style="color: #ef4444; font-weight: bold;">java</code>
                        <p>Java Collections/APIs</p>
                    </div>
                </div>
            </div>
        `;
    } else {
        div.innerHTML = '';
    }
}

// Render benchmarks
function renderBenchmarks() {
    const container = document.getElementById('benchmarks');
    if (!container) return;
    
    container.innerHTML = benchmarks.map(bench => `
        <div class="benchmark">
            <div class="benchmark-header">
                <span class="benchmark-name">${bench.name}</span>
                <span class="benchmark-value">${bench.value}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${bench.percent}%"></div>
            </div>
        </div>
    `).join('');
}

// Set active section
function setActiveSection(section) {
    activeSection = section;
    renderNavButtons();
    renderSectionContent();
    renderEmbeddingHighlight();
}

// Set active example
function setActiveExample(example) {
    activeExample = example;
    renderExampleTabs();
    renderExampleCode();
}

// Initialize page
function init() {
    try {
        // Create particles
        createParticles();
        
        // Initial renders
        updateHeroCode();
        renderFeatures();
        renderSyntaxExamples();
        renderNavButtons();
        renderSectionContent();
        renderExampleTabs();
        renderExampleCode();
        renderEmbeddingHighlight();
        renderBenchmarks();
        
        // Auto-rotate hero code
        setInterval(updateHeroCode, 4000);
    } catch (error) {
        console.error('Initialization error:', error);
    }
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Smooth scroll for navigation links
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});