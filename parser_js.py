# parser_js.py
"""
JavaScript/TypeScript AST parsing using Tree-Sitter.
Extracts functions, classes, calls, and imports from JS/TS/JSX/TSX files.

Supports:
- ES6 modules (import/export)
- CommonJS (require/module.exports)
- Arrow functions, function declarations, class methods
- JSX/TSX components
- TypeScript type annotations
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser
    JS_LANGUAGE = Language(tsjavascript.language())
    JS_AVAILABLE = True
except ImportError:
    JS_AVAILABLE = False
    logger.warning("tree-sitter-javascript not installed; JS parsing disabled")

try:
    import tree_sitter_typescript as tstypescript
    TS_LANGUAGE = Language(tstypescript.language_typescript())
    TSX_LANGUAGE = Language(tstypescript.language_tsx())
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    logger.warning("tree-sitter-typescript not installed; TS parsing disabled")

# Import shared dataclasses from parser.py
from parser import (
    FunctionDef, ClassDef, ImportRef, CallEdge, VariableRef, ParsedFile
)


# ---------------------------------------------------------------------------
# JavaScript-specific helpers
# ---------------------------------------------------------------------------

def _get_node_text(node, source_bytes: bytes) -> str:
    """Extract text from a tree-sitter node."""
    if node is None:
        return ""
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _extract_js_docstring(node, source_bytes: bytes) -> Optional[str]:
    """
    Extract JSDoc comment or first comment before the function/class.
    
    Example:
        /**
         * Validates user credentials
         * @param {string} username
         * @returns {User}
         */
        function validateUser(username) { ... }
    """
    # Look for comment nodes before this node
    prev_sibling = node.prev_sibling
    while prev_sibling:
        if prev_sibling.type == 'comment':
            comment_text = _get_node_text(prev_sibling, source_bytes)
            # Clean up JSDoc formatting
            if comment_text.startswith('/**') and comment_text.endswith('*/'):
                # Extract first line of JSDoc
                lines = comment_text[3:-2].strip().split('\n')
                for line in lines:
                    line = line.strip().lstrip('*').strip()
                    if line and not line.startswith('@'):
                        return line
            elif comment_text.startswith('//'):
                return comment_text[2:].strip()
        prev_sibling = prev_sibling.prev_sibling
    
    return None


def _build_js_fqn(module_name: str, class_stack: List[str], func_name: str) -> str:
    """
    Build fully qualified name for JavaScript function.
    
    Examples:
        module: "src/components/Button", class: ["Button"], func: "render"
        → "src.components.Button.render"
        
        module: "src/utils/auth", class: [], func: "validateUser"
        → "src.utils.auth.validateUser"
    """
    parts = [module_name.replace('/', '.').replace('\\', '.')]
    parts.extend(class_stack)
    parts.append(func_name)
    return '.'.join(parts)


def _extract_js_params(params_node, source_bytes: bytes) -> List[str]:
    """
    Extract parameter names from formal_parameters node.
    
    Handles:
    - Simple params: (a, b, c)
    - Default params: (a = 1, b = 2)
    - Destructuring: ({x, y}, [a, b])
    - Rest params: (...args)
    - TypeScript types: (a: string, b: number)
    """
    if params_node is None:
        return []
    
    params = []
    for child in params_node.children:
        if child.type == 'identifier':
            params.append(_get_node_text(child, source_bytes))
        elif child.type == 'required_parameter':
            # TypeScript: param: type
            pattern = child.child_by_field_name('pattern')
            if pattern:
                params.append(_get_node_text(pattern, source_bytes))
        elif child.type == 'optional_parameter':
            # TypeScript: param?: type
            pattern = child.child_by_field_name('pattern')
            if pattern:
                params.append(_get_node_text(pattern, source_bytes) + '?')
        elif child.type == 'rest_parameter':
            # ...args
            pattern = child.child_by_field_name('pattern')
            if pattern:
                params.append('...' + _get_node_text(pattern, source_bytes))
        elif child.type == 'assignment_pattern':
            # Default parameter: a = 1
            left = child.child_by_field_name('left')
            if left:
                params.append(_get_node_text(left, source_bytes))
        elif child.type in ('object_pattern', 'array_pattern'):
            # Destructuring: {x, y} or [a, b]
            params.append(_get_node_text(child, source_bytes))
    
    return params


def _extract_js_return_type(node, source_bytes: bytes) -> Optional[str]:
    """
    Extract TypeScript return type annotation.
    
    Example:
        function foo(): string { ... }
        → "string"
    """
    type_annotation = node.child_by_field_name('return_type')
    if type_annotation:
        # Skip the ':' token
        for child in type_annotation.children:
            if child.type != ':':
                return _get_node_text(child, source_bytes)
    return None


# ---------------------------------------------------------------------------
# JavaScript function extraction
# ---------------------------------------------------------------------------

def _extract_js_function(
    node,
    source_bytes: bytes,
    file_path: str,
    module_name: str,
    class_stack: List[str],
    is_method: bool = False
) -> Optional[FunctionDef]:
    """
    Extract a FunctionDef from a JavaScript function node.
    
    Handles:
    - function_declaration: function foo() {}
    - method_definition: class { foo() {} }
    - arrow_function: const foo = () => {}
    - function_expression: const foo = function() {}
    """
    # Get function name
    name_node = node.child_by_field_name('name')
    if name_node:
        func_name = _get_node_text(name_node, source_bytes)
    else:
        # Arrow function or anonymous function
        # Try to get name from parent assignment
        parent = node.parent
        if parent and parent.type == 'variable_declarator':
            name_node = parent.child_by_field_name('name')
            if name_node:
                func_name = _get_node_text(name_node, source_bytes)
            else:
                func_name = '<anonymous>'
        else:
            func_name = '<anonymous>'
    
    # Get parameters
    params_node = node.child_by_field_name('parameters')
    params = _extract_js_params(params_node, source_bytes)
    
    # Get return type (TypeScript)
    return_annotation = _extract_js_return_type(node, source_bytes)
    
    # Get docstring
    docstring = _extract_js_docstring(node, source_bytes)
    
    # Build FQN
    qualname = _build_js_fqn(module_name, class_stack, func_name)
    
    # Get class name if method
    class_name = class_stack[-1] if class_stack else None
    
    return FunctionDef(
        name=func_name,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        docstring=docstring,
        is_method=is_method,
        class_name=class_name,
        qualname=qualname,
        params=params,
        decorators=[],  # JavaScript doesn't have decorators (except TypeScript experimental)
        return_annotation=return_annotation,
        raises=[],  # Will be filled by call extraction
    )


# ---------------------------------------------------------------------------
# JavaScript class extraction
# ---------------------------------------------------------------------------

def _extract_js_class(
    node,
    source_bytes: bytes,
    file_path: str,
    module_name: str,
    class_stack: List[str]
) -> Optional[ClassDef]:
    """
    Extract a ClassDef from a JavaScript class node.
    
    Handles:
    - class_declaration: class Foo extends Bar {}
    - class_expression: const Foo = class extends Bar {}
    """
    # Get class name
    name_node = node.child_by_field_name('name')
    if name_node:
        class_name = _get_node_text(name_node, source_bytes)
    else:
        # Anonymous class expression
        parent = node.parent
        if parent and parent.type == 'variable_declarator':
            name_node = parent.child_by_field_name('name')
            if name_node:
                class_name = _get_node_text(name_node, source_bytes)
            else:
                class_name = '<anonymous>'
        else:
            class_name = '<anonymous>'
    
    # Get base classes (extends clause)
    bases = []
    heritage_node = node.child_by_field_name('heritage')
    if heritage_node:
        for child in heritage_node.children:
            if child.type in ('identifier', 'member_expression'):
                bases.append(_get_node_text(child, source_bytes))
    
    # Get docstring
    docstring = _extract_js_docstring(node, source_bytes)
    
    # Build qualname
    qualname = _build_js_fqn(module_name, class_stack, class_name)
    
    return ClassDef(
        name=class_name,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        docstring=docstring,
        qualname=qualname,
        bases=bases,
    )


# ---------------------------------------------------------------------------
# JavaScript call extraction
# ---------------------------------------------------------------------------

def _extract_js_call(
    node,
    source_bytes: bytes,
    file_path: str,
    caller_qualname: str
) -> Optional[CallEdge]:
    """
    Extract a CallEdge from a JavaScript call_expression node.
    
    Handles:
    - Simple calls: foo()
    - Method calls: obj.method()
    - Chained calls: obj.method1().method2()
    - Optional chaining: obj?.method?.()
    - Constructor calls: new Foo()
    """
    func_node = node.child_by_field_name('function')
    if func_node is None:
        return None
    
    # Get the full call expression
    callee_expr = _get_node_text(func_node, source_bytes)
    
    # Get column of the method name (for jedi-like resolution)
    if func_node.type == 'member_expression':
        # obj.method → column of 'method'
        property_node = func_node.child_by_field_name('property')
        if property_node:
            column = property_node.start_point[1]
        else:
            column = func_node.start_point[1]
    else:
        # Simple call → column of function name
        column = func_node.start_point[1]
    
    return CallEdge(
        caller_qualname=caller_qualname,
        callee_expr=callee_expr,
        file_path=file_path,
        line=node.start_point[0] + 1,
        column=column,
    )


# ---------------------------------------------------------------------------
# JavaScript import/export extraction
# ---------------------------------------------------------------------------

def _extract_js_imports(node, source_bytes: bytes, file_path: str) -> List[ImportRef]:
    """
    Extract ImportRef objects from JavaScript import statements.
    
    Handles:
    - ES6 imports: import { x, y } from 'module'
    - ES6 default: import x from 'module'
    - ES6 namespace: import * as x from 'module'
    - CommonJS: const x = require('module')
    """
    imports = []
    
    if node.type == 'import_statement':
        # ES6 import
        source_node = node.child_by_field_name('source')
        if source_node:
            # Remove quotes from module name
            module = _get_node_text(source_node, source_bytes).strip('"\'')
            
            # Check for different import types
            for child in node.children:
                if child.type == 'import_clause':
                    # import { x, y } from 'module'
                    for subchild in child.children:
                        if subchild.type == 'named_imports':
                            # { x, y }
                            for spec in subchild.children:
                                if spec.type == 'import_specifier':
                                    name_node = spec.child_by_field_name('name')
                                    alias_node = spec.child_by_field_name('alias')
                                    if name_node:
                                        imported_name = _get_node_text(name_node, source_bytes)
                                        alias = _get_node_text(alias_node, source_bytes) if alias_node else None
                                        imports.append(ImportRef(
                                            file_path=file_path,
                                            module=module,
                                            alias=alias,
                                            imported_name=imported_name,
                                        ))
                        elif subchild.type == 'identifier':
                            # import x from 'module' (default import)
                            alias = _get_node_text(subchild, source_bytes)
                            imports.append(ImportRef(
                                file_path=file_path,
                                module=module,
                                alias=alias,
                                imported_name='default',
                            ))
                        elif subchild.type == 'namespace_import':
                            # import * as x from 'module'
                            for ns_child in subchild.children:
                                if ns_child.type == 'identifier':
                                    alias = _get_node_text(ns_child, source_bytes)
                                    imports.append(ImportRef(
                                        file_path=file_path,
                                        module=module,
                                        alias=alias,
                                        imported_name='*',
                                    ))
    
    elif node.type == 'variable_declaration':
        # Check for CommonJS require: const x = require('module')
        for declarator in node.children:
            if declarator.type == 'variable_declarator':
                value_node = declarator.child_by_field_name('value')
                if value_node and value_node.type == 'call_expression':
                    func_node = value_node.child_by_field_name('function')
                    if func_node and _get_node_text(func_node, source_bytes) == 'require':
                        # This is a require() call
                        args_node = value_node.child_by_field_name('arguments')
                        if args_node:
                            for arg in args_node.children:
                                if arg.type == 'string':
                                    module = _get_node_text(arg, source_bytes).strip('"\'')
                                    name_node = declarator.child_by_field_name('name')
                                    if name_node:
                                        alias = _get_node_text(name_node, source_bytes)
                                        imports.append(ImportRef(
                                            file_path=file_path,
                                            module=module,
                                            alias=alias,
                                            imported_name=None,  # CommonJS doesn't have named imports
                                        ))
    
    return imports


def _extract_js_exports(node, source_bytes: bytes) -> List[str]:
    """
    Extract exported names from JavaScript export statements.
    
    Handles:
    - export { x, y }
    - export default x
    - export function foo() {}
    - export class Bar {}
    - module.exports = x (CommonJS)
    """
    exports = []
    
    if node.type == 'export_statement':
        # ES6 export
        for child in node.children:
            if child.type == 'export_clause':
                # export { x, y }
                for spec in child.children:
                    if spec.type == 'export_specifier':
                        name_node = spec.child_by_field_name('name')
                        if name_node:
                            exports.append(_get_node_text(name_node, source_bytes))
            elif child.type in ('function_declaration', 'class_declaration'):
                # export function foo() {} or export class Bar {}
                name_node = child.child_by_field_name('name')
                if name_node:
                    exports.append(_get_node_text(name_node, source_bytes))
            elif child.type == 'lexical_declaration':
                # export const x = ...
                for declarator in child.children:
                    if declarator.type == 'variable_declarator':
                        name_node = declarator.child_by_field_name('name')
                        if name_node:
                            exports.append(_get_node_text(name_node, source_bytes))
    
    elif node.type == 'expression_statement':
        # Check for CommonJS: module.exports = x
        expr = node.child_by_field_name('expression')
        if expr and expr.type == 'assignment_expression':
            left = expr.child_by_field_name('left')
            if left and left.type == 'member_expression':
                obj = left.child_by_field_name('object')
                prop = left.child_by_field_name('property')
                if (obj and _get_node_text(obj, source_bytes) == 'module' and
                    prop and _get_node_text(prop, source_bytes) == 'exports'):
                    # This is module.exports = ...
                    # Mark as default export
                    exports.append('default')
    
    return exports


# ---------------------------------------------------------------------------
# Main JavaScript AST walker
# ---------------------------------------------------------------------------

def _walk_js_tree(
    node,
    result: ParsedFile,
    source_bytes: bytes,
    file_path: str,
    module_name: str,
    class_stack: List[str],
    func_stack: List[str],
) -> None:
    """
    Recursively walk the JavaScript AST, extracting functions, classes, calls, and imports.
    
    Similar to Python's _walk_tree but handles JavaScript-specific constructs.
    """
    # Extract classes
    if node.type in ('class_declaration', 'class'):
        class_def = _extract_js_class(node, source_bytes, file_path, module_name, class_stack)
        if class_def:
            result.classes.append(class_def)
            # Recurse into class body with updated class stack
            new_class_stack = class_stack + [class_def.name]
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    _walk_js_tree(child, result, source_bytes, file_path, module_name,
                                new_class_stack, func_stack)
            return  # Don't process children again
    
    # Extract functions
    if node.type in ('function_declaration', 'function', 'arrow_function', 'function_expression'):
        is_method = len(class_stack) > 0
        func_def = _extract_js_function(node, source_bytes, file_path, module_name,
                                       class_stack, is_method)
        if func_def:
            result.functions.append(func_def)
            # Recurse into function body with updated func stack
            new_func_stack = func_stack + [func_def.name]
            body_node = node.child_by_field_name('body')
            if body_node:
                # Extract calls within this function
                caller_qualname = func_def.qualname
                for child in body_node.named_children:
                    _extract_js_calls_recursive(child, result, source_bytes, file_path, caller_qualname)
            return  # Don't process children again
    
    # Extract method definitions (class methods)
    if node.type == 'method_definition':
        is_method = True
        func_def = _extract_js_function(node, source_bytes, file_path, module_name,
                                       class_stack, is_method)
        if func_def:
            result.functions.append(func_def)
            # Extract calls within method
            body_node = node.child_by_field_name('body')
            if body_node:
                caller_qualname = func_def.qualname
                for child in body_node.named_children:
                    _extract_js_calls_recursive(child, result, source_bytes, file_path, caller_qualname)
            return
    
    # Extract imports
    if node.type in ('import_statement', 'variable_declaration'):
        imports = _extract_js_imports(node, source_bytes, file_path)
        result.imports.extend(imports)
    
    # Extract exports
    if node.type in ('export_statement', 'expression_statement'):
        exports = _extract_js_exports(node, source_bytes)
        result.exports.extend(exports)
    
    # Recurse into children
    for child in node.named_children:
        _walk_js_tree(child, result, source_bytes, file_path, module_name,
                     class_stack, func_stack)


def _extract_js_calls_recursive(
    node,
    result: ParsedFile,
    source_bytes: bytes,
    file_path: str,
    caller_qualname: str
) -> None:
    """Recursively extract all call expressions from a node."""
    if node.type == 'call_expression':
        call = _extract_js_call(node, source_bytes, file_path, caller_qualname)
        if call:
            result.calls.append(call)
    
    # Recurse into children
    for child in node.named_children:
        _extract_js_calls_recursive(child, result, source_bytes, file_path, caller_qualname)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_js_file(file_path: str, source_code: str) -> ParsedFile:
    """
    Parse a JavaScript/TypeScript file and extract all code entities.
    
    Args:
        file_path: Relative path to the file (e.g., "src/components/Button.js")
        source_code: Full source code as string
    
    Returns:
        ParsedFile with functions, classes, imports, calls, and exports
    
    Raises:
        ValueError: If tree-sitter-javascript/typescript not installed
        SyntaxError: If the file has syntax errors (returns partial result)
    """
    # Check if parsers are available
    ext = Path(file_path).suffix.lower()
    if ext in ('.js', '.jsx'):
        if not JS_AVAILABLE:
            raise ValueError("tree-sitter-javascript not installed")
        language = JS_LANGUAGE
    elif ext in ('.ts', '.tsx'):
        if not TS_AVAILABLE:
            raise ValueError("tree-sitter-typescript not installed")
        language = TSX_LANGUAGE if ext == '.tsx' else TS_LANGUAGE
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Parse the source code
    parser = Parser(language)
    source_bytes = source_code.encode('utf-8')
    tree = parser.parse(source_bytes)
    
    # Check for syntax errors
    if tree.root_node.has_error:
        logger.warning(f"Syntax errors in {file_path}, returning partial result")
    
    # Initialize result
    module_name = file_path.replace('\\', '/').replace('.js', '').replace('.jsx', '').replace('.ts', '').replace('.tsx', '')
    result = ParsedFile(file_path=file_path)
    
    # Walk the AST
    _walk_js_tree(
        tree.root_node,
        result,
        source_bytes,
        file_path,
        module_name,
        class_stack=[],
        func_stack=[],
    )
    
    logger.info(f"Parsed {file_path}: {len(result.functions)} functions, "
                f"{len(result.classes)} classes, {len(result.imports)} imports, "
                f"{len(result.calls)} calls")
    
    return result


# ---------------------------------------------------------------------------
# Test/Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    print("Testing JavaScript Parser...")
    print("-" * 60)
    
    # Test 1: Simple React component
    test_code_react = """
import React from 'react';

/**
 * Button component for user interactions
 */
export class Button extends React.Component {
  handleClick() {
    this.props.onClick();
  }
  
  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}

export const createButton = (props) => {
  return new Button(props);
};
"""
    
    print("\n1. Testing React component parsing...")
    result = parse_js_file("src/components/Button.jsx", test_code_react)
    print(f"✓ Functions: {len(result.functions)}")
    for func in result.functions:
        print(f"  - {func.qualname} (method: {func.is_method})")
    print(f"✓ Classes: {len(result.classes)}")
    for cls in result.classes:
        print(f"  - {cls.qualname} extends {cls.bases}")
    print(f"✓ Imports: {len(result.imports)}")
    for imp in result.imports:
        print(f"  - {imp.module} ({imp.imported_name or 'default'})")
    print(f"✓ Calls: {len(result.calls)}")
    for call in result.calls[:3]:
        print(f"  - {call.caller_qualname} → {call.callee_expr}")
    
    # Test 2: Node.js module with CommonJS
    test_code_node = """
const express = require('express');
const { validateUser } = require('./auth');

function createServer(port) {
  const app = express();
  
  app.post('/login', (req, res) => {
    const user = validateUser(req.body.username, req.body.password);
    res.json({ user });
  });
  
  return app;
}

module.exports = createServer;
"""
    
    print("\n2. Testing Node.js module parsing...")
    result = parse_js_file("src/server.js", test_code_node)
    print(f"✓ Functions: {len(result.functions)}")
    for func in result.functions:
        print(f"  - {func.qualname}")
    print(f"✓ Imports: {len(result.imports)}")
    for imp in result.imports:
        print(f"  - {imp.module} as {imp.alias}")
    print(f"✓ Exports: {len(result.exports)}")
    print(f"  - {result.exports}")
    
    print("\n" + "-" * 60)
    print("JavaScript parser test complete!")

# Made with Bob
