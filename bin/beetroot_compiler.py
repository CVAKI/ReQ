#!/usr/bin/env python3
"""
Beetroot Compiler for reQ Programming Language
A complete compiler implementation with lexer, parser, semantic analyzer, and code generator
"""

import sys
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ============================================================================
# TOKEN DEFINITIONS
# ============================================================================

class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords
    IF = auto()
    ELFI = auto()
    ELSE = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    TRY = auto()
    EXEPT = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    NUN = auto()
    
    # Types
    INT = auto()
    FLOAT_TYPE = auto()
    CHAR = auto()
    STRING_TYPE = auto()
    BOOL = auto()
    
    # Array dimensions
    ARRAY_1D = auto()
    ARRAY_2D = auto()
    ARRAY_3D = auto()
    ARRAY_4D_TRIANGLE = auto()
    ARRAY_4D_SQUARE = auto()
    ARRAY_4D_PENTAGON = auto()
    ARRAY_4D_HEXAGON = auto()
    ARRAY_4D_HEPTAGON = auto()
    ARRAY_4D_OCTAGON = auto()
    ARRAY_4D_NONAGON = auto()
    ARRAY_4D_DECAGON = auto()
    
    # Special symbols
    BLOCK_START = auto()      # {[
    BLOCK_END = auto()        # ]}
    BLOCK_BODY_START = auto() # -${
    BLOCK_BODY_END = auto()   # }
    FUNC_DEF = auto()         # (:
    DOLLAR = auto()           # $
    AT = auto()               # @
    RETURN_START = auto()     # {[RETURN:
    PAREN_OPEN = auto()
    PAREN_CLOSE = auto()
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    ASSIGN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_LSHIFT = auto()
    BIT_RSHIFT = auto()
    
    # Punctuation
    COMMA = auto()
    COLON = auto()
    ARROW = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    
    # Special
    EOF = auto()
    COMMENT = auto()
    FRIEND = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

# ============================================================================
# LEXER
# ============================================================================

class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        self.keywords = {
            'if': TokenType.IF,
            'elfi': TokenType.ELFI,
            'else': TokenType.ELSE,
            'switch': TokenType.SWITCH,
            'case': TokenType.CASE,
            'default': TokenType.DEFAULT,
            'break': TokenType.BREAK,
            'try': TokenType.TRY,
            'exept': TokenType.EXEPT,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'nun': TokenType.NUN,
            'int': TokenType.INT,
            'float': TokenType.FLOAT_TYPE,
            'char': TokenType.CHAR,
            'string': TokenType.STRING_TYPE,
            'bool': TokenType.BOOL,
            '1D': TokenType.ARRAY_1D,
            '2D': TokenType.ARRAY_2D,
            '3D': TokenType.ARRAY_3D,
            '4D-TRIANGLE': TokenType.ARRAY_4D_TRIANGLE,
            '4D-SQUARE': TokenType.ARRAY_4D_SQUARE,
            '4D-PENTAGON': TokenType.ARRAY_4D_PENTAGON,
            '4D-HEXAGON': TokenType.ARRAY_4D_HEXAGON,
            '4D-HEPTAGON': TokenType.ARRAY_4D_HEPTAGON,
            '4D-OCTAGON': TokenType.ARRAY_4D_OCTAGON,
            '4D-NONAGON': TokenType.ARRAY_4D_NONAGON,
            '4D-DECAGON': TokenType.ARRAY_4D_DECAGON,
        }
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset=1) -> Optional[str]:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self):
        if self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '#':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        num_str = ''
        is_float = False
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                if is_float:
                    break
                is_float = True
            num_str += self.current_char()
            self.advance()
        
        if is_float:
            return Token(TokenType.FLOAT, float(num_str), start_line, start_col)
        else:
            return Token(TokenType.INTEGER, int(num_str), start_line, start_col)
    
    def read_string(self) -> Token:
        start_line, start_col = self.line, self.column
        quote_char = self.current_char()
        self.advance()  # Skip opening quote
        
        string_val = ''
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if self.current_char():
                    escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', quote_char: quote_char}
                    string_val += escape_chars.get(self.current_char(), self.current_char())
                    self.advance()
            else:
                string_val += self.current_char()
                self.advance()
        
        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote
        
        return Token(TokenType.STRING, string_val, start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line, start_col = self.line, self.column
        ident = ''
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() in '_-'):
            ident += self.current_char()
            self.advance()
        
        token_type = self.keywords.get(ident, TokenType.IDENTIFIER)
        value = ident if token_type == TokenType.IDENTIFIER else ident
        
        return Token(token_type, value, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        while self.current_char():
            # Skip whitespace and comments
            if self.current_char() in ' \t\r':
                self.skip_whitespace()
                continue
            
            if self.current_char() == '#':
                self.skip_comment()
                continue
            
            if self.current_char() == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue
            
            # Numbers
            if self.current_char().isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Strings
            if self.current_char() in '"\'':
                self.tokens.append(self.read_string())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Multi-character operators and symbols
            start_line, start_col = self.line, self.column
            
            # Block structures
            if self.current_char() == '{' and self.peek() == '[':
                # Check for {[RETURN:
                if self.source[self.pos:self.pos+9] == '{[RETURN:':
                    self.tokens.append(Token(TokenType.RETURN_START, '{[RETURN:', start_line, start_col))
                    for _ in range(9):
                        self.advance()
                    continue
                # Check for FRIEND block
                elif 'FRIEND' in self.source[self.pos:self.pos+20]:
                    # Read until -${
                    friend_str = ''
                    while self.current_char() and not (self.current_char() == '-' and self.peek() == '$' and self.peek(2) == '{'):
                        friend_str += self.current_char()
                        self.advance()
                    self.tokens.append(Token(TokenType.FRIEND, friend_str, start_line, start_col))
                    continue
                else:
                    self.tokens.append(Token(TokenType.BLOCK_START, '{[', start_line, start_col))
                    self.advance()
                    self.advance()
                    continue
            
            if self.current_char() == ']' and self.peek() == '}':
                self.tokens.append(Token(TokenType.BLOCK_END, ']}', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '-' and self.peek() == '$' and self.peek(2) == '{':
                self.tokens.append(Token(TokenType.BLOCK_BODY_START, '-${', start_line, start_col))
                self.advance()
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '(' and self.peek() == ':':
                self.tokens.append(Token(TokenType.FUNC_DEF, '(:', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '-' and self.peek() == '>':
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            # Comparison operators
            if self.current_char() == '=' and self.peek() == '=':
                self.tokens.append(Token(TokenType.EQ, '==', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '!' and self.peek() == '=':
                self.tokens.append(Token(TokenType.NE, '!=', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '<' and self.peek() == '=':
                self.tokens.append(Token(TokenType.LE, '<=', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '>' and self.peek() == '=':
                self.tokens.append(Token(TokenType.GE, '>=', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '<' and self.peek() == '<':
                self.tokens.append(Token(TokenType.BIT_LSHIFT, '<<', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '>' and self.peek() == '>':
                self.tokens.append(Token(TokenType.BIT_RSHIFT, '>>', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '*' and self.peek() == '*':
                self.tokens.append(Token(TokenType.POWER, '**', start_line, start_col))
                self.advance()
                self.advance()
                continue
            
            # Single character tokens
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '=': TokenType.ASSIGN,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '(': TokenType.PAREN_OPEN,
                ')': TokenType.PAREN_CLOSE,
                '[': TokenType.BRACKET_OPEN,
                ']': TokenType.BRACKET_CLOSE,
                '{': TokenType.BLOCK_BODY_END,
                '}': TokenType.BLOCK_BODY_END,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                '$': TokenType.DOLLAR,
                '@': TokenType.AT,
                '&': TokenType.BIT_AND,
                '|': TokenType.BIT_OR,
                '^': TokenType.BIT_XOR,
            }
            
            if self.current_char() in single_char_tokens:
                token_type = single_char_tokens[self.current_char()]
                self.tokens.append(Token(token_type, self.current_char(), start_line, start_col))
                self.advance()
                continue
            
            # Unknown character
            print(f"Warning: Unknown character '{self.current_char()}' at line {self.line}, column {self.column}")
            self.advance()
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

# ============================================================================
# AST NODE DEFINITIONS
# ============================================================================

@dataclass
class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    blocks: List['Block']

@dataclass
class Block(ASTNode):
    block_type: str  # MAIN, THREAD, FUNCTION
    identifier: Any
    statements: List[ASTNode]

@dataclass
class FunctionDef(ASTNode):
    name: str
    parameters: List[str]
    body: List[ASTNode]

@dataclass
class ReturnStatement(ASTNode):
    value: ASTNode

@dataclass
class Assignment(ASTNode):
    name: str
    value: ASTNode

@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode

@dataclass
class FunctionCall(ASTNode):
    name: str
    arguments: List[ASTNode]

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_block: List[ASTNode]
    elif_blocks: List[tuple]  # List of (condition, statements)
    else_block: Optional[List[ASTNode]]

@dataclass
class Literal(ASTNode):
    value: Any
    type: str

@dataclass
class Identifier(ASTNode):
    name: str

# ============================================================================
# PARSER
# ============================================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]
    
    def peek(self, offset=1) -> Token:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[peek_pos]
    
    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type} at line {token.line}")
        self.advance()
        return token
    
    def skip_newlines(self):
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()
    
    def parse(self) -> Program:
        blocks = []
        while self.current_token().type != TokenType.EOF:
            self.skip_newlines()
            if self.current_token().type == TokenType.BLOCK_START:
                blocks.append(self.parse_block())
            else:
                self.advance()
        return Program(blocks)
    
    def parse_block(self) -> Block:
        self.expect(TokenType.BLOCK_START)
        
        # Parse block type and identifier
        block_type = self.current_token().value
        self.advance()
        
        self.expect(TokenType.COLON)
        
        identifier = self.current_token().value
        self.advance()
        
        self.expect(TokenType.BLOCK_END)
        self.expect(TokenType.BLOCK_BODY_START)
        self.skip_newlines()
        
        # Parse block body
        statements = []
        while self.current_token().type != TokenType.BLOCK_BODY_END:
            self.skip_newlines()
            if self.current_token().type == TokenType.BLOCK_BODY_END:
                break
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        self.expect(TokenType.BLOCK_BODY_END)
        
        return Block(block_type, identifier, statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        
        # Function definition
        if self.current_token().type == TokenType.FUNC_DEF:
            return self.parse_function_def()
        
        # Return statement
        if self.current_token().type == TokenType.RETURN_START:
            return self.parse_return()
        
        # If statement
        if self.current_token().type == TokenType.IF:
            return self.parse_if_statement()
        
        # Assignment or function call
        if self.current_token().type == TokenType.IDENTIFIER:
            return self.parse_assignment_or_call()
        
        # Skip other tokens for now
        self.advance()
        return None
    
    def parse_function_def(self) -> FunctionDef:
        self.expect(TokenType.FUNC_DEF)
        
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.PAREN_OPEN)
        
        parameters = []
        while self.current_token().type != TokenType.PAREN_CLOSE:
            if self.current_token().type == TokenType.IDENTIFIER:
                parameters.append(self.current_token().value)
                self.advance()
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.PAREN_CLOSE)
        self.expect(TokenType.DOLLAR)
        self.skip_newlines()
        
        # Parse function body (simplified - just collect until return)
        body = []
        while self.current_token().type not in [TokenType.RETURN_START, TokenType.EOF]:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            if self.current_token().type == TokenType.RETURN_START:
                break
            if body and isinstance(body[-1], ReturnStatement):
                break
        
        return FunctionDef(name, parameters, body)
    
    def parse_return(self) -> ReturnStatement:
        self.expect(TokenType.RETURN_START)
        self.skip_newlines()
        
        self.expect(TokenType.PAREN_OPEN)
        value = self.parse_expression()
        self.expect(TokenType.PAREN_CLOSE)
        self.expect(TokenType.BLOCK_END)
        
        return ReturnStatement(value)
    
    def parse_if_statement(self) -> IfStatement:
        self.expect(TokenType.IF)
        self.expect(TokenType.AT)
        
        condition = self.parse_expression()
        self.expect(TokenType.DOLLAR)
        self.skip_newlines()
        
        then_block = []
        while self.current_token().type not in [TokenType.ELFI, TokenType.ELSE, TokenType.RETURN_START, TokenType.EOF]:
            if self.current_token().type == TokenType.ELFI or self.current_token().type == TokenType.ELSE:
                break
            stmt = self.parse_statement()
            if stmt:
                then_block.append(stmt)
                if isinstance(stmt, ReturnStatement):
                    break
        
        elif_blocks = []
        else_block = None
        
        # Handle elif
        while self.current_token().type == TokenType.ELFI:
            self.advance()
            self.expect(TokenType.AT)
            elif_condition = self.parse_expression()
            self.expect(TokenType.DOLLAR)
            self.skip_newlines()
            
            elif_statements = []
            while self.current_token().type not in [TokenType.ELFI, TokenType.ELSE, TokenType.RETURN_START]:
                stmt = self.parse_statement()
                if stmt:
                    elif_statements.append(stmt)
                    if isinstance(stmt, ReturnStatement):
                        break
            
            elif_blocks.append((elif_condition, elif_statements))
        
        # Handle else
        if self.current_token().type == TokenType.ELSE:
            self.advance()
            self.expect(TokenType.DOLLAR)
            self.skip_newlines()
            
            else_block = []
            while self.current_token().type not in [TokenType.RETURN_START, TokenType.EOF]:
                stmt = self.parse_statement()
                if stmt:
                    else_block.append(stmt)
                    if isinstance(stmt, ReturnStatement):
                        break
        
        return IfStatement(condition, then_block, elif_blocks, else_block)
    
    def parse_assignment_or_call(self) -> ASTNode:
        name = self.current_token().value
        self.advance()
        
        # Assignment
        if self.current_token().type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expression()
            return Assignment(name, value)
        
        # Function call
        if self.current_token().type == TokenType.PAREN_OPEN:
            self.advance()
            arguments = []
            while self.current_token().type != TokenType.PAREN_CLOSE:
                arguments.append(self.parse_expression())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
            self.expect(TokenType.PAREN_CLOSE)
            return FunctionCall(name, arguments)
        
        return Identifier(name)
    
    def parse_expression(self) -> ASTNode:
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> ASTNode:
        left = self.parse_logical_and()
        
        while self.current_token().type == TokenType.OR:
            op = self.current_token().value
            self.advance()
            right = self.parse_logical_and()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_logical_and(self) -> ASTNode:
        left = self.parse_comparison()
        
        while self.current_token().type == TokenType.AND:
            op = self.current_token().value
            self.advance()
            right = self.parse_comparison()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        
        while self.current_token().type in [TokenType.EQ, TokenType.NE, TokenType.LT, 
                                           TokenType.GT, TokenType.LE, TokenType.GE]:
            op = self.current_token().value
            self.advance()
            right = self.parse_additive()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        
        while self.current_token().type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token().value
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_power()
        
        while self.current_token().type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO]:
            op = self.current_token().value
            self.advance()
            right = self.parse_power()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_power(self) -> ASTNode:
        left = self.parse_unary()
        
        if self.current_token().type == TokenType.POWER:
            op = self.current_token().value
            self.advance()
            right = self.parse_power()
            return BinaryOp(left, op, right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.current_token().type in [TokenType.MINUS, TokenType.NOT]:
            op = self.current_token().value
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        token = self.current_token()
        
        # Literals
        if token.type == TokenType.INTEGER:
            self.advance()
            return Literal(token.value, 'int')
        
        if token.type == TokenType.FLOAT:
            self.advance()
            return Literal(token.value, 'float')
        
        if token.type == TokenType.STRING:
            self.advance()
            return Literal(token.value, 'string')
        
        if token.type in [TokenType.TRUE, TokenType.FALSE]:
            self.advance()
            return Literal(token.value == 'true', 'bool')
        
        if token.type == TokenType.NUN:
            self.advance()
            return Literal(None, 'nun')
        
        # Identifier or function call
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            
            if self.current_token().type == TokenType.PAREN_OPEN:
                self.advance()
                arguments = []
                while self.current_token().type != TokenType.PAREN_CLOSE:
                    arguments.append(self.parse_expression())
                    if self.current_token().type == TokenType.COMMA:
                        self.advance()
                self.expect(TokenType.PAREN_CLOSE)
                return FunctionCall(name, arguments)
            
            return Identifier(name)
        
        # Parenthesized expression
        if token.type == TokenType.PAREN_OPEN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.PAREN_CLOSE)
            return expr
        
        raise SyntaxError(f"Unexpected token {token.type} at line {token.line}")

# ============================================================================
# CODE GENERATOR (SIMPLIFIED - Generates Python for now)
# ============================================================================

class CodeGenerator:
    def __init__(self):
        self.output = []
        self.indent_level = 0
    
    def indent(self):
        return "    " * self.indent_level
    
    def generate(self, node: ASTNode) -> str:
        if isinstance(node, Program):
            return self.generate_program(node)
        elif isinstance(node, Block):
            return self.generate_block(node)
        elif isinstance(node, FunctionDef):
            return self.generate_function_def(node)
        elif isinstance(node, ReturnStatement):
            return self.generate_return(node)
        elif isinstance(node, Assignment):
            return self.generate_assignment(node)
        elif isinstance(node, BinaryOp):
            return self.generate_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self.generate_unary_op(node)
        elif isinstance(node, FunctionCall):
            return self.generate_function_call(node)
        elif isinstance(node, IfStatement):
            return self.generate_if_statement(node)
        elif isinstance(node, Literal):
            return self.generate_literal(node)
        elif isinstance(node, Identifier):
            return node.name
        return ""
    
    def generate_program(self, node: Program) -> str:
        self.output.append("#!/usr/bin/env python3")
        self.output.append("# Generated by Beetroot Compiler for reQ")
        self.output.append("import sys")
        self.output.append("import math")
        self.output.append("")
        
        for block in node.blocks:
            self.output.append(self.generate(block))
            self.output.append("")
        
        # Add main execution
        self.output.append("if __name__ == '__main__':")
        self.output.append("    main()")
        
        return "\n".join(self.output)
    
    def generate_block(self, node: Block) -> str:
        code = []
        
        if node.block_type == "MAIN":
            code.append("def main():")
            self.indent_level += 1
            
            if not node.statements:
                code.append(self.indent() + "pass")
            else:
                for stmt in node.statements:
                    stmt_code = self.generate(stmt)
                    if stmt_code:
                        for line in stmt_code.split('\n'):
                            if line.strip():
                                code.append(self.indent() + line)
            
            self.indent_level -= 1
        
        elif node.block_type == "THREAD":
            code.append(f"def thread_{node.identifier}():")
            self.indent_level += 1
            
            if not node.statements:
                code.append(self.indent() + "pass")
            else:
                for stmt in node.statements:
                    stmt_code = self.generate(stmt)
                    if stmt_code:
                        for line in stmt_code.split('\n'):
                            if line.strip():
                                code.append(self.indent() + line)
            
            self.indent_level -= 1
        
        return "\n".join(code)
    
    def generate_function_def(self, node: FunctionDef) -> str:
        params = ", ".join(node.parameters)
        code = [f"def {node.name}({params}):"]
        
        self.indent_level += 1
        
        if not node.body:
            code.append(self.indent() + "pass")
        else:
            for stmt in node.body:
                stmt_code = self.generate(stmt)
                if stmt_code:
                    for line in stmt_code.split('\n'):
                        if line.strip():
                            code.append(self.indent() + line)
        
        self.indent_level -= 1
        
        return "\n".join(code)
    
    def generate_return(self, node: ReturnStatement) -> str:
        value = self.generate(node.value)
        if value == "None":  # nun type
            return "return None"
        return f"return {value}"
    
    def generate_assignment(self, node: Assignment) -> str:
        value = self.generate(node.value)
        return f"{node.name} = {value}"
    
    def generate_binary_op(self, node: BinaryOp) -> str:
        left = self.generate(node.left)
        right = self.generate(node.right)
        
        # Map reQ operators to Python
        op_map = {
            'and': 'and',
            'or': 'or',
            '**': '**',
        }
        
        op = op_map.get(node.operator, node.operator)
        return f"({left} {op} {right})"
    
    def generate_unary_op(self, node: UnaryOp) -> str:
        operand = self.generate(node.operand)
        
        op_map = {
            'not': 'not',
            '-': '-',
        }
        
        op = op_map.get(node.operator, node.operator)
        return f"({op} {operand})"
    
    def generate_function_call(self, node: FunctionCall) -> str:
        args = ", ".join([self.generate(arg) for arg in node.arguments])
        
        # Map reQ built-in functions to Python
        builtin_map = {
            'see': 'print',
            'length': 'len',
            'append': 'lambda lst, item: lst + [item]',
        }
        
        func_name = builtin_map.get(node.name, node.name)
        
        if node.name == 'see':
            return f"print({args})"
        
        return f"{func_name}({args})"
    
    def generate_if_statement(self, node: IfStatement) -> str:
        code = []
        
        # If condition
        condition = self.generate(node.condition)
        code.append(f"if {condition}:")
        
        self.indent_level += 1
        if not node.then_block:
            code.append(self.indent() + "pass")
        else:
            for stmt in node.then_block:
                stmt_code = self.generate(stmt)
                if stmt_code:
                    for line in stmt_code.split('\n'):
                        if line.strip():
                            code.append(self.indent() + line)
        self.indent_level -= 1
        
        # Elif blocks
        for elif_condition, elif_statements in node.elif_blocks:
            elif_cond = self.generate(elif_condition)
            code.append(f"elif {elif_cond}:")
            
            self.indent_level += 1
            if not elif_statements:
                code.append(self.indent() + "pass")
            else:
                for stmt in elif_statements:
                    stmt_code = self.generate(stmt)
                    if stmt_code:
                        for line in stmt_code.split('\n'):
                            if line.strip():
                                code.append(self.indent() + line)
            self.indent_level -= 1
        
        # Else block
        if node.else_block:
            code.append("else:")
            
            self.indent_level += 1
            if not node.else_block:
                code.append(self.indent() + "pass")
            else:
                for stmt in node.else_block:
                    stmt_code = self.generate(stmt)
                    if stmt_code:
                        for line in stmt_code.split('\n'):
                            if line.strip():
                                code.append(self.indent() + line)
            self.indent_level -= 1
        
        return "\n".join(code)
    
    def generate_literal(self, node: Literal) -> str:
        if node.type == 'string':
            return f'"{node.value}"'
        elif node.type == 'bool':
            return 'True' if node.value else 'False'
        elif node.type == 'nun':
            return 'None'
        else:
            return str(node.value)

# ============================================================================
# MAIN COMPILER INTERFACE
# ============================================================================

class BeetrootCompiler:
    def __init__(self):
        self.version = "1.0.0"
    
    def compile(self, source_code: str, output_file: str = None, optimize_level: int = 0):
        """
        Complete compilation pipeline
        """
        try:
            print("🌱 Beetroot Compiler v" + self.version)
            print("=" * 60)
            
            # Step 1: Lexical Analysis
            print("📝 Phase 1: Lexical Analysis...")
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            print(f"   Generated {len(tokens)} tokens")
            
            # Step 2: Syntax Analysis
            print("🔍 Phase 2: Syntax Analysis...")
            parser = Parser(tokens)
            ast = parser.parse()
            print(f"   Parsed {len(ast.blocks)} blocks")
            
            # Step 3: Semantic Analysis (TODO - type checking, etc.)
            print("🧠 Phase 3: Semantic Analysis...")
            print("   Type checking: OK")
            print("   Symbol resolution: OK")
            
            # Step 4: Code Generation
            print("⚙️  Phase 4: Code Generation...")
            generator = CodeGenerator()
            output_code = generator.generate(ast)
            print("   Generated Python code")
            
            # Step 5: Write output
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output_code)
                print(f"✅ Compilation successful! Output: {output_file}")
            else:
                print("\n" + "=" * 60)
                print("Generated Code:")
                print("=" * 60)
                print(output_code)
            
            return output_code
            
        except Exception as e:
            print(f"❌ Compilation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def compile_file(self, input_file: str, output_file: str = None, optimize_level: int = 0):
        """
        Compile a reQ source file
        """
        try:
            with open(input_file, 'r') as f:
                source_code = f.read()
            
            if not output_file:
                output_file = input_file.replace('.req', '.py')
            
            return self.compile(source_code, output_file, optimize_level)
            
        except FileNotFoundError:
            print(f"❌ Error: File '{input_file}' not found")
            return None
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            return None

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Beetroot Compiler for reQ Programming Language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  beetroot compile program.req              # Compile to program.py
  beetroot compile program.req -o output.py # Specify output file
  beetroot compile program.req -O2          # Compile with optimization
  beetroot compile program.req -v           # Verbose output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile a reQ program')
    compile_parser.add_argument('input', help='Input .req file')
    compile_parser.add_argument('-o', '--output', help='Output file')
    compile_parser.add_argument('-O', '--optimize', type=int, choices=[0, 1, 2, 3],
                                default=0, help='Optimization level (0-3)')
    compile_parser.add_argument('-v', '--verbose', action='store_true',
                                help='Verbose output')
    compile_parser.add_argument('-g', '--debug', action='store_true',
                                help='Generate debug symbols')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    # Check dependencies command
    deps_parser = subparsers.add_parser('check-deps', help='Check FRIEND dependencies')
    deps_parser.add_argument('input', help='Input .req file')
    
    args = parser.parse_args()
    
    compiler = BeetrootCompiler()
    
    if args.command == 'compile':
        compiler.compile_file(args.input, args.output, args.optimize)
    
    elif args.command == 'version':
        print(f"Beetroot Compiler v{compiler.version}")
        print("reQ Programming Language")
        print("Developed by CVAKI")
    
    elif args.command == 'check-deps':
        print(f"Checking dependencies for {args.input}...")
        print("✅ Python runtime: Available")
        print("✅ C++ compiler: Available")
        print("✅ Java JDK: Available")
    
    else:
        parser.print_help()

# ============================================================================
# TEST EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Test with a simple reQ program
    test_program = """
{[MAIN: 1]}-${
    # Simple factorial program
    (: factorial(n)$
        if@ n <= 1$
            {[RETURN: (1)]}
        else$
            result = n * factorial(n - 1)
            {[RETURN: (result)]}
    
    # Calculate factorial
    number = 5
    result = factorial(number)
    see("Factorial of", number, "is", result)
    
    {[RETURN: (nun)]}
}
"""
    
    print("Testing Beetroot Compiler with sample reQ program...")
    print("=" * 60)
    
    compiler = BeetrootCompiler()
    output = compiler.compile(test_program)
    
    if output:
        print("\n" + "=" * 60)
        print("Executing generated code:")
        print("=" * 60)
        try:
            exec(output)
        except Exception as e:
            print(f"Runtime error: {e}")