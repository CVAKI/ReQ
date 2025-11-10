#!/usr/bin/env python3
"""
FRIEND System (Foreign Runtime Integration and Embedded Native Development)
Handles embedding and execution of Python, C++, and Java code within reQ
"""

import os
import subprocess
import tempfile
import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ============================================================================
# FRIEND LANGUAGE TYPES
# ============================================================================

class FriendLanguage(Enum):
    PYTHON = "py"
    CPP = "cpp"
    JAVA = "java"

# ============================================================================
# FRIEND BLOCK PARSER
# ============================================================================

@dataclass
class FriendBlock:
    language: FriendLanguage
    code: str
    return_type: Optional[str]
    imports: List[str]
    line_number: int

class FriendParser:
    """
    Extracts and parses FRIEND blocks from reQ source code
    """
    
    def __init__(self):
        self.blocks = []
    
    def extract_friend_blocks(self, source_code: str) -> List[FriendBlock]:
        """
        Extract all FRIEND blocks from reQ source
        """
        self.blocks = []
        
        # Pattern to match FRIEND blocks
        pattern = r'\{FRIEND\["(\w+)"\]\}-\$\{(.*?)\{?\[RETURN:\s*\((.*?)\)\]\}'
        
        matches = re.finditer(pattern, source_code, re.DOTALL)
        
        for match in matches:
            language_str = match.group(1)
            code = match.group(2).strip()
            return_expr = match.group(3).strip()
            
            # Determine language
            lang = None
            if language_str == "py":
                lang = FriendLanguage.PYTHON
            elif language_str == "cpp":
                lang = FriendLanguage.CPP
            elif language_str == "java":
                lang = FriendLanguage.JAVA
            
            if lang:
                block = FriendBlock(
                    language=lang,
                    code=code,
                    return_type=None,
                    imports=[],
                    line_number=source_code[:match.start()].count('\n') + 1
                )
                self.blocks.append(block)
        
        return self.blocks

# ============================================================================
# PYTHON FRIEND EXECUTOR
# ============================================================================

class PythonFriendExecutor:
    """
    Executes embedded Python code
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def execute(self, block: FriendBlock, context: Dict[str, Any] = None) -> Any:
        """
        Execute Python FRIEND block
        """
        # Create temporary Python file
        temp_file = os.path.join(self.temp_dir, f"friend_py_{id(block)}.py")
        
        # Build complete Python script
        script = self._build_python_script(block, context)
        
        # Write to file
        with open(temp_file, 'w') as f:
            f.write(script)
        
        # Execute
        try:
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output
                output = result.stdout.strip()
                return self._parse_output(output)
            else:
                raise RuntimeError(f"Python execution failed: {result.stderr}")
        
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _build_python_script(self, block: FriendBlock, context: Dict[str, Any]) -> str:
        """
        Build a complete Python script from FRIEND block
        """
        script_parts = []
        
        # Add imports
        script_parts.append("import json")
        script_parts.append("import sys")
        script_parts.append("")
        
        # Add context variables (marshaled from reQ)
        if context:
            script_parts.append("# Variables from reQ")
            for var_name, var_value in context.items():
                script_parts.append(f"{var_name}_from_req = {json.dumps(var_value)}")
            script_parts.append("")
        
        # Add user code
        script_parts.append("# User FRIEND code")
        script_parts.append(block.code)
        script_parts.append("")
        
        return "\n".join(script_parts)
    
    def _parse_output(self, output: str) -> Any:
        """
        Parse Python output back to reQ type
        """
        try:
            # Try to parse as JSON
            return json.loads(output)
        except:
            # Return as string
            return output

# ============================================================================
# C++ FRIEND EXECUTOR
# ============================================================================

class CppFriendExecutor:
    """
    Compiles and executes embedded C++ code
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.compiler = 'g++'  # or 'clang++'
    
    def execute(self, block: FriendBlock, context: Dict[str, Any] = None) -> Any:
        """
        Compile and execute C++ FRIEND block
        """
        # Create temporary C++ file
        temp_cpp = os.path.join(self.temp_dir, f"friend_cpp_{id(block)}.cpp")
        temp_exe = os.path.join(self.temp_dir, f"friend_cpp_{id(block)}")
        
        # Build complete C++ program
        program = self._build_cpp_program(block, context)
        
        # Write to file
        with open(temp_cpp, 'w') as f:
            f.write(program)
        
        try:
            # Compile
            compile_result = subprocess.run(
                [self.compiler, '-std=c++17', '-O2', temp_cpp, '-o', temp_exe],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                raise RuntimeError(f"C++ compilation failed: {compile_result.stderr}")
            
            # Execute
            exec_result = subprocess.run(
                [temp_exe],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if exec_result.returncode == 0:
                return self._parse_output(exec_result.stdout.strip())
            else:
                raise RuntimeError(f"C++ execution failed: {exec_result.stderr}")
        
        finally:
            # Cleanup
            for f in [temp_cpp, temp_exe]:
                if os.path.exists(f):
                    os.remove(f)
    
    def _build_cpp_program(self, block: FriendBlock, context: Dict[str, Any]) -> str:
        """
        Build a complete C++ program from FRIEND block
        """
        program_parts = []
        
        # Standard includes
        program_parts.append("#include <iostream>")
        program_parts.append("#include <string>")
        program_parts.append("#include <vector>")
        program_parts.append("")
        
        # Extract additional includes from user code
        user_includes = re.findall(r'#include\s+[<"].*?[>"]', block.code)
        if user_includes:
            program_parts.extend(user_includes)
            program_parts.append("")
        
        # Remove includes from user code
        user_code = re.sub(r'#include\s+[<"].*?[>"]', '', block.code)
        
        # Main function
        program_parts.append("int main() {")
        
        # Add context variables
        if context:
            program_parts.append("    // Variables from reQ")
            for var_name, var_value in context.items():
                cpp_value = self._marshal_to_cpp(var_value)
                program_parts.append(f"    auto {var_name}_from_req = {cpp_value};")
            program_parts.append("")
        
        # Add user code (indented)
        for line in user_code.split('\n'):
            if line.strip():
                program_parts.append("    " + line)
        
        program_parts.append("    return 0;")
        program_parts.append("}")
        
        return "\n".join(program_parts)
    
    def _marshal_to_cpp(self, value: Any) -> str:
        """
        Convert Python/reQ value to C++ code
        """
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f"{value}f"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, list):
            elements = ", ".join([self._marshal_to_cpp(v) for v in value])
            return f"{{{elements}}}"
        else:
            return str(value)
    
    def _parse_output(self, output: str) -> Any:
        """
        Parse C++ output back to reQ type
        """
        try:
            # Try numeric conversion
            if '.' in output:
                return float(output)
            else:
                return int(output)
        except:
            return output

# ============================================================================
# JAVA FRIEND EXECUTOR
# ============================================================================

class JavaFriendExecutor:
    """
    Compiles and executes embedded Java code
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.compiler = 'javac'
    
    def execute(self, block: FriendBlock, context: Dict[str, Any] = None) -> Any:
        """
        Compile and execute Java FRIEND block
        """
        # Create temporary Java file
        class_name = f"FriendJava_{id(block)}"
        temp_java = os.path.join(self.temp_dir, f"{class_name}.java")
        
        # Build complete Java program
        program = self._build_java_program(block, context, class_name)
        
        # Write to file
        with open(temp_java, 'w') as f:
            f.write(program)
        
        try:
            # Compile
            compile_result = subprocess.run(
                [self.compiler, temp_java],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.temp_dir
            )
            
            if compile_result.returncode != 0:
                raise RuntimeError(f"Java compilation failed: {compile_result.stderr}")
            
            # Execute
            exec_result = subprocess.run(
                ['java', '-cp', self.temp_dir, class_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if exec_result.returncode == 0:
                return self._parse_output(exec_result.stdout.strip())
            else:
                raise RuntimeError(f"Java execution failed: {exec_result.stderr}")
        
        finally:
            # Cleanup
            for ext in ['.java', '.class']:
                f = os.path.join(self.temp_dir, f"{class_name}{ext}")
                if os.path.exists(f):
                    os.remove(f)
    
    def _build_java_program(self, block: FriendBlock, context: Dict[str, Any], class_name: str) -> str:
        """
        Build a complete Java program from FRIEND block
        """
        program_parts = []
        
        # Extract imports from user code
        user_imports = re.findall(r'import\s+[\w.]+\*?;', block.code)
        if user_imports:
            program_parts.extend(user_imports)
            program_parts.append("")
        
        # Remove imports from user code
        user_code = re.sub(r'import\s+[\w.]+\*?;', '', block.code)
        
        # Class declaration
        program_parts.append(f"public class {class_name} {{")
        program_parts.append("    public static void main(String[] args) {")
        
        # Add context variables
        if context:
            program_parts.append("        // Variables from reQ")
            for var_name, var_value in context.items():
                java_value = self._marshal_to_java(var_value)
                java_type = self._get_java_type(var_value)
                program_parts.append(f"        {java_type} {var_name}_from_req = {java_value};")
            program_parts.append("")
        
        # Add user code (indented)
        for line in user_code.split('\n'):
            if line.strip():
                program_parts.append("        " + line)
        
        program_parts.append("    }")
        program_parts.append("}")
        
        return "\n".join(program_parts)
    
    def _marshal_to_java(self, value: Any) -> str:
        """
        Convert Python/reQ value to Java code
        """
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f"{value}d"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, list):
            elements = ", ".join([self._marshal_to_java(v) for v in value])
            return f"Arrays.asList({elements})"
        else:
            return str(value)
    
    def _get_java_type(self, value: Any) -> str:
        """
        Determine Java type from Python value
        """
        if isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, list):
            if value and isinstance(value[0], int):
                return "List<Integer>"
            elif value and isinstance(value[0], float):
                return "List<Double>"
            else:
                return "List<Object>"
        return "Object"
    
    def _parse_output(self, output: str) -> Any:
        """
        Parse Java output back to reQ type
        """
        try:
            if '.' in output:
                return float(output)
            else:
                return int(output)
        except:
            return output

# ============================================================================
# FRIEND MANAGER
# ============================================================================

class FriendManager:
    """
    Manages FRIEND system: parsing, compilation, and execution
    """
    
    def __init__(self):
        self.parser = FriendParser()
        self.executors = {
            FriendLanguage.PYTHON: PythonFriendExecutor(),
            FriendLanguage.CPP: CppFriendExecutor(),
            FriendLanguage.JAVA: JavaFriendExecutor(),
        }
    
    def check_runtime_availability(self) -> Dict[FriendLanguage, bool]:
        """
        Check which language runtimes are available
        """
        availability = {}
        
        # Check Python
        try:
            result = subprocess.run(['python3', '--version'], capture_output=True, timeout=5)
            availability[FriendLanguage.PYTHON] = result.returncode == 0
        except:
            availability[FriendLanguage.PYTHON] = False
        
        # Check C++
        try:
            result = subprocess.run(['g++', '--version'], capture_output=True, timeout=5)
            availability[FriendLanguage.CPP] = result.returncode == 0
        except:
            availability[FriendLanguage.CPP] = False
        
        # Check Java
        try:
            result = subprocess.run(['javac', '-version'], capture_output=True, timeout=5)
            availability[FriendLanguage.JAVA] = result.returncode == 0
        except:
            availability[FriendLanguage.JAVA] = False
        
        return availability
    
    def extract_blocks(self, source_code: str) -> List[FriendBlock]:
        """
        Extract all FRIEND blocks from source
        """
        return self.parser.extract_friend_blocks(source_code)
    
    def execute_block(self, block: FriendBlock, context: Dict[str, Any] = None) -> Any:
        """
        Execute a single FRIEND block
        """
        executor = self.executors.get(block.language)
        if not executor:
            raise RuntimeError(f"No executor for language {block.language}")
        
        return executor.execute(block, context)
    
    def generate_interface_code(self, blocks: List[FriendBlock]) -> str:
        """
        Generate interface code for FRIEND blocks
        """
        interface_code = []
        
        interface_code.append("# FRIEND Block Interfaces")
        interface_code.append("# Auto-generated by Beetroot Compiler")
        interface_code.append("")
        
        for i, block in enumerate(blocks):
            interface_code.append(f"def friend_block_{i}(**context):")
            interface_code.append(f"    # Language: {block.language.value}")
            interface_code.append(f"    # Line: {block.line_number}")
            interface_code.append("    pass")
            interface_code.append("")
        
        return "\n".join(interface_code)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("FRIEND System - Multi-Language Integration")
    print("=" * 60)
    
    manager = FriendManager()
    
    # Check runtime availability
    print("\n🔍 Checking language runtime availability...")
    availability = manager.check_runtime_availability()
    
    for lang, available in availability.items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"   {lang.value.upper()}: {status}")
    
    print("\n📋 FRIEND System supports:")
    print("   • Python: ML, data science, rapid prototyping")
    print("   • C++: Performance-critical code, system programming")
    print("   • Java: Enterprise APIs, Android development")
    
    print("\n💡 Example FRIEND block:")
    print("""
    result = {FRIEND["py"]}-${
        import numpy as np
        data = np.random.randn(100)
        {[RETURN: (data.mean())]}
    }
    """)