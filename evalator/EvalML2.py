import sys
import re
from dataclasses import dataclass, field
from typing import List, Union, Tuple

# -------------------------------------------------------------------
# 1. 式の構造 (AST)
# -------------------------------------------------------------------
class Expression:
    def to_string(self) -> str: return str(self)
@dataclass
class IntLiteral(Expression):
    value: int
    def __str__(self): return str(self.value)
@dataclass
class BoolLiteral(Expression):
    value: bool
    def __str__(self): return str(self.value).lower()
@dataclass
class Variable(Expression):
    name: str
    def __str__(self): return self.name
precedence = {'LessThan': 1, 'Plus': 2, 'Minus': 2, 'Times': 3}
@dataclass
class BinaryOp(Expression):
    op: str; left: Expression; right: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, -1)
    def to_string(self) -> str:
        my_prec = self.get_precedence()
        left_str = self.left.to_string()
        if isinstance(self.left, BinaryOp) and self.left.get_precedence() < my_prec: left_str = f"({left_str})"
        right_str = self.right.to_string()
        if isinstance(self.right, BinaryOp) and self.right.get_precedence() <= my_prec: right_str = f"({right_str})"
        return f"{left_str} {self.op} {right_str}"
class Plus(BinaryOp):
    def __init__(self, left, right): super().__init__('+', left, right)
class Minus(BinaryOp):
    def __init__(self, left, right): super().__init__('-', left, right)
class Times(BinaryOp):
    def __init__(self, left, right): super().__init__('*', left, right)
class LessThan(BinaryOp):
    def __init__(self, left, right): super().__init__('<', left, right)
@dataclass
class If(Expression):
    cond: Expression; true_branch: Expression; false_branch: Expression
    def __str__(self): return f"if {self.cond} then {self.true_branch} else {self.false_branch}"
@dataclass
class Let(Expression):
    var: str; bound_expr: Expression; body_expr: Expression
    def __str__(self): return f"let {self.var} = {self.bound_expr} in {self.body_expr}"
Value = Union[int, bool]
Environment = List[Tuple[str, Value]]
Token = Tuple[str, str]

# -------------------------------------------------------------------
# 2. 導出規則 (Derivation)
# -------------------------------------------------------------------
def format_env(env: Environment) -> str:
    if not env: return ""
    return ", ".join(f"{var} = {str(val).lower() if isinstance(val, bool) else val}" for var, val in env) + " "

class Derivation:
    def format(self, indent_level=0) -> str: raise NotImplementedError
@dataclass
class BPlus(Derivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} plus {self.n2} is {self.result} by B-Plus {{}};'
@dataclass
class BMinus(Derivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} minus {self.n2} is {self.result} by B-Minus {{}};'
@dataclass
class BTimes(Derivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} times {self.n2} is {self.result} by B-Times {{}};'
@dataclass
class BLt(Derivation):
    n1: int; n2: int; result: bool
    def format(self, i=0): return f'{" "*i}{self.n1} less than {self.n2} is {str(self.result).lower()} by B-Lt {{}};'
@dataclass
class EInt(Derivation):
    env: Environment; value: int
    def format(self, i=0): return f'{" "*i}{format_env(self.env)}|- {self.value} evalto {self.value} by E-Int {{}};'
@dataclass
class EBool(Derivation):
    env: Environment; value: bool
    def format(self, i=0): return f'{" "*i}{format_env(self.env)}|- {str(self.value).lower()} evalto {str(self.value).lower()} by E-Bool {{}};'
@dataclass
class EBinOp(Derivation):
    env: Environment; expr: Expression; value: Value; premises: List[Derivation]; rule_name: str
    def format(self, i=0) -> str:
        indent, val_str = " " * i, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = "\n".join(p.format(i + 1) for p in self.premises)
        return f"{indent}{format_env(self.env)}|- {self.expr} evalto {val_str} by {self.rule_name} {{\n{premise_str}\n{indent}}};"
@dataclass
class EIf(Derivation):
    env: Environment; expr: Expression; value: Value; premises: List[Derivation]; rule_name: str
    def format(self, i=0) -> str:
        indent, val_str = " " * i, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = "\n".join(p.format(i + 1) for p in self.premises)
        return f"{indent}{format_env(self.env)}|- {self.expr} evalto {val_str} by {self.rule_name} {{\n{premise_str}\n{indent}}};"
@dataclass
class ELet(Derivation):
    env: Environment; expr: Expression; value: Value; premises: List[Derivation]
    def format(self, i=0) -> str:
        indent, val_str = " " * i, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = "\n".join(p.format(i + 1) for p in self.premises)
        return f"{indent}{format_env(self.env)}|- {self.expr} evalto {val_str} by E-Let {{\n{premise_str}\n{indent}}};"
@dataclass
class EVar1(Derivation):
    env: Environment; var_name: str; value: Value
    def format(self, i=0) -> str:
        val_str = str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        return f'{" "*i}{format_env(self.env)}|- {self.var_name} evalto {val_str} by E-Var1 {{}};'
@dataclass
class EVar2(Derivation):
    env: Environment; var_name: str; value: Value; premise: Derivation
    def format(self, i=0) -> str:
        indent, val_str = " " * i, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = self.premise.format(i + 1)
        return f"{indent}{format_env(self.env)}|- {self.var_name} evalto {val_str} by E-Var2 {{\n{premise_str}\n{indent}}};"

# -------------------------------------------------------------------
# 3. Parser 
# -------------------------------------------------------------------
class Parser:
    def __init__(self, tokens: List[Token]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        keywords = {'if', 'then', 'else', 'true', 'false', 'let', 'in'}
        token_spec = [
            ('ID',       r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('INT',      r'-?\d+'),
            ('OP',       r'[+*<()-=]'),
            ('SKIP',     r'\s+'),
            ('MISMATCH', r'.'),
        ]
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_spec)
        tokens: List[Token] = []
        for mo in re.finditer(tok_regex, text):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            elif kind == 'MISMATCH': raise SyntaxError(f"認識できない文字です: '{value}'")
            if kind == 'ID' and value in keywords: kind = value.upper() # Treat keywords as their own kind
            tokens.append((kind, value))
        return cls(tokens)

    def current_token(self) -> Token | None: return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def consume(self, expected_val=None):
        if expected_val and (self.current_token() is None or self.current_token()[1] != expected_val):
             raise SyntaxError(f"'{expected_val}'が予期されましたが、'{self.current_token()}'が見つかりました")
        self.pos += 1
        
    def parse_expression(self) -> Expression:
        kind, _ = self.current_token() if self.current_token() else (None, None)
        if kind == 'LET': return self.parse_let()
        if kind == 'IF': return self.parse_if()
        return self.parse_comparison()

    def parse_let(self) -> Expression:
        self.consume('let'); _, var_name = self.current_token(); self.consume(var_name); self.consume('=')
        bound_expr = self.parse_expression(); self.consume('in'); body_expr = self.parse_expression()
        return Let(var_name, bound_expr, body_expr)
        
    def parse_if(self) -> Expression:
        self.consume('if'); cond = self.parse_expression(); self.consume('then')
        true_branch = self.parse_expression(); self.consume('else'); false_branch = self.parse_expression()
        return If(cond, true_branch, false_branch)

    def parse_comparison(self) -> Expression:
        node = self.parse_add_sub()
        if self.current_token() and self.current_token()[1] == '<':
            self.consume('<'); right_node = self.parse_add_sub(); node = LessThan(node, right_node)
        return node
        
    def parse_add_sub(self) -> Expression:
        node = self.parse_mul()
        while self.current_token() and self.current_token()[1] in ['+', '-']:
            _, op_val = self.current_token()
            self.consume(op_val); right_node = self.parse_mul(); node = Plus(node, right_node) if op_val == '+' else Minus(node, right_node)
        return node
        
    def parse_mul(self) -> Expression:
        node = self.parse_primary()
        while self.current_token() and self.current_token()[1] == '*':
            self.consume('*'); right_node = self.parse_primary(); node = Times(node, right_node)
        return node
        
    def parse_primary(self) -> Expression:
        if self.current_token() is None: raise SyntaxError("式の途中で入力が終了しました")
        kind, val = self.current_token()
        if kind == 'INT': self.consume(val); return IntLiteral(int(val))
        if kind == 'ID': self.consume(val); return Variable(val)
        if kind == 'TRUE': self.consume(val); return BoolLiteral(True)
        if kind == 'FALSE': self.consume(val); return BoolLiteral(False)
        if val == '(': self.consume('('); node = self.parse_expression(); self.consume(')'); return node
        raise SyntaxError(f"予期しないトークンです: '{val}'")

def run_parser_on_text(text: str) -> Expression:
    if not text.strip(): raise SyntaxError("式が空です")
    parser = Parser.from_text(text)
    node = parser.parse_expression()
    if parser.current_token() is not None: raise SyntaxError(f"解析完了後に余分なトークンがあります: '{parser.current_token()}'")
    return node

# -------------------------------------------------------------------
# 4. 評価器 (Evaluator)
# -------------------------------------------------------------------
def evaluate(env: Environment, node: Expression) -> tuple[Value, Derivation]:
    if isinstance(node, IntLiteral): return node.value, EInt(env, node.value)
    if isinstance(node, BoolLiteral): return node.value, EBool(env, node.value)
    if isinstance(node, Variable): return lookup_var_in_env(env, node.name)
    if isinstance(node, Let):
        v1, d1 = evaluate(env, node.bound_expr); new_env = env + [(node.var, v1)]; v2, d2 = evaluate(new_env, node.body_expr)
        return v2, ELet(env, node, v2, [d1, d2])
    if isinstance(node, If):
        cond_val, cond_deriv = evaluate(env, node.cond)
        if not isinstance(cond_val, bool): raise TypeError("Ifの条件はbool値であるべきです")
        branch_to_eval = node.true_branch if cond_val else node.false_branch; val, branch_deriv = evaluate(env, branch_to_eval)
        return val, EIf(env, node, val, [cond_deriv, branch_deriv], "E-IfT" if cond_val else "E-IfF")
    if isinstance(node, BinaryOp):
        val1, deriv1 = evaluate(env, node.left); val2, deriv2 = evaluate(env, node.right)
        if not (isinstance(val1, int) and isinstance(val2, int)):
            if not (isinstance(node, LessThan) and isinstance(val1, int) and isinstance(val2, int)): raise TypeError(f"'{node.op}'演算子は整数にのみ適用できます")
        if isinstance(node, Plus): result, b_deriv, name = val1 + val2, BPlus(val1, val2, val1 + val2), "E-Plus"
        elif isinstance(node, Minus): result, b_deriv, name = val1 - val2, BMinus(val1, val2, val1 - val2), "E-Minus"
        elif isinstance(node, Times): result, b_deriv, name = val1 * val2, BTimes(val1, val2, val1 * val2), "E-Times"
        elif isinstance(node, LessThan): result, b_deriv, name = val1 < val2, BLt(val1, val2, val1 < val2), "E-Lt"
        else: raise TypeError("不明な二項演算子です")
        return result, EBinOp(env, node, result, [deriv1, deriv2, b_deriv], name)
    raise TypeError(f"不明な式の型です: {type(node)}")

def lookup_var_in_env(env: Environment, name: str) -> tuple[Value, Derivation]:
    if not env: raise NameError(f"未定義の変数です: '{name}'")
    last_var, last_val = env[-1]; rest_env = env[:-1]
    if last_var == name: return last_val, EVar1(env, name, last_val)
    else:
        val_in_rest, premise_deriv = lookup_var_in_env(rest_env, name)
        return val_in_rest, EVar2(env, name, val_in_rest, premise_deriv)

# -------------------------------------------------------------------
# 5. メイン実行部 - ★★★ 修正箇所 ★★★
# -------------------------------------------------------------------
def parse_env_str(env_str: str) -> Environment:
    if not env_str.strip(): return []
    env: Environment = []
    # 修正1: reversed()を削除し、入力通りの順で環境を構築
    for binding in env_str.split(','):
        var, val_str = [p.strip() for p in binding.split('=')]
        if val_str == 'true': value = True
        elif val_str == 'false': value = False
        else: value = int(val_str)
        env.append((var, value))
    return env

def main():
    print("導出したい式を入力してください。例: x=1|-x+1 evalto 2")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            parts = line.split("|-", 1)
            if len(parts) == 2: env_str, expr_part = parts
            else: env_str, expr_part = "", parts[0]
            expr_str = expr_part.split(" evalto ", 1)[0].strip()
            try:
                initial_env = parse_env_str(env_str)
                ast = run_parser_on_text(expr_str)
                _, derivation_tree = evaluate(initial_env, ast)
                formatted_tree = derivation_tree.format(0)
                if formatted_tree.endswith(';'): formatted_tree = formatted_tree[:-1]
                print(formatted_tree)
                print("-" * 20)
            except (SyntaxError, TypeError, ValueError, NameError) as e: print(f"エラー: {e}", file=sys.stderr)
    except KeyboardInterrupt: print("\n終了します。")

if __name__ == "__main__":
    main()