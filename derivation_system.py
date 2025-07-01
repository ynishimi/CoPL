# derivation_system.py (最終修正版)
import sys
import re
from dataclasses import dataclass, field
from typing import List, Union

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
precedence = {'LessThan': 1, 'Plus': 2, 'Minus': 2, 'Times': 3, 'If': 0}
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
Value = Union[int, bool]

# -------------------------------------------------------------------
# 2. 導出規則 (Derivation)
# -------------------------------------------------------------------
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
    value: int
    def format(self, i=0): return f'{" "*i}{self.value} evalto {self.value} by E-Int {{}};'
@dataclass
class EBool(Derivation):
    value: bool
    def format(self, i=0): return f'{" "*i}{str(self.value).lower()} evalto {str(self.value).lower()} by E-Bool {{}};'
@dataclass
class EBinOp(Derivation):
    expr: Expression; value: Value; premises: List[Derivation]; rule_name: str
    def format(self, indent_level=0) -> str:
        indent, val_str = " " * indent_level, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = "\n".join(p.format(indent_level + 1) for p in self.premises)
        last_char = "" if indent_level == 0 else ";"
        return f"{indent}{self.expr} evalto {val_str} by {self.rule_name} {{\n{premise_str}\n{indent}}}{last_char}"
@dataclass
class EIf(Derivation):
    expr: Expression; value: Value; premises: List[Derivation]; rule_name: str
    def format(self, indent_level=0) -> str:
        indent, val_str = " " * indent_level, str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
        premise_str = "\n".join(p.format(indent_level + 1) for p in self.premises)
        last_char = "" if indent_level == 0 else ";"
        return f"{indent}{self.expr} evalto {val_str} by {self.rule_name} {{\n{premise_str}\n{indent}}}{last_char}"

# -------------------------------------------------------------------
# 3. パーサー (Parser) - ★★★ ifの優先順位を修正 ★★★
# -------------------------------------------------------------------
class Parser:
    def __init__(self, tokens: List[str]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        token_spec = [('KEYWORD', r'\b(if|then|else|true|false)\b'), ('INT', r'-?\d+'), ('OP', r'[+*<()-]'), ('SKIP', r'\s+'), ('MISMATCH', r'.')]
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, text):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            elif kind == 'MISMATCH': raise SyntaxError(f"認識できない文字です: '{value}'")
            tokens.append(value)
        return cls(tokens)

    def current_token(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self, expected=None):
        if expected and self.current_token() != expected: raise SyntaxError(f"'{expected}'が予期されましたが、'{self.current_token()}'が見つかりました")
        self.pos += 1

    def parse_expression(self) -> Expression:
        return self.parse_if() # ★修正: 解析は常にif(最低優先順位)から試みる

    def parse_if(self) -> Expression:
        if self.current_token() == 'if':
            self.consume('if'); cond = self.parse_expression(); self.consume('then'); true_branch = self.parse_expression(); self.consume('else'); false_branch = self.parse_expression()
            return If(cond, true_branch, false_branch)
        return self.parse_comparison()

    def parse_comparison(self) -> Expression:
        node = self.parse_add_sub()
        if self.current_token() == '<': self.consume('<'); right_node = self.parse_add_sub(); node = LessThan(node, right_node)
        return node
    def parse_add_sub(self) -> Expression:
        node = self.parse_mul()
        while self.current_token() in ['+', '-']: op = self.current_token(); self.consume(op); right_node = self.parse_mul(); node = Plus(node, right_node) if op == '+' else Minus(node, right_node)
        return node
    def parse_mul(self) -> Expression:
        node = self.parse_primary()
        while self.current_token() == '*': self.consume('*'); right_node = self.parse_primary(); node = Times(node, right_node)
        return node
    def parse_primary(self) -> Expression:
        token = self.current_token()
        if token is None: raise SyntaxError("式の途中で入力が終了しました")
        # ★修正: ifの解析ロジックをここから削除
        if token.lstrip('-').isdigit(): self.consume(); return IntLiteral(int(token))
        if token == 'true': self.consume(); return BoolLiteral(True)
        if token == 'false': self.consume(); return BoolLiteral(False)
        if token == '(': self.consume('('); node = self.parse_expression(); self.consume(')'); return node
        raise SyntaxError(f"予期しないトークンです: '{token}'")

def run_parser_on_text(text: str) -> Expression:
    if not text.strip(): raise SyntaxError("式が空です")
    parser = Parser.from_text(text)
    node = parser.parse_expression()
    if parser.current_token() is not None: raise SyntaxError(f"解析完了後に余分なトークンがあります: '{parser.current_token()}'")
    return node

# -------------------------------------------------------------------
# 4. 評価器 (Evaluator)
# -------------------------------------------------------------------
def evaluate(node: Expression) -> tuple[Value, Derivation]:
    if isinstance(node, IntLiteral): return node.value, EInt(node.value)
    if isinstance(node, BoolLiteral): return node.value, EBool(node.value)
    if isinstance(node, If):
        cond_val, cond_deriv = evaluate(node.cond)
        if not isinstance(cond_val, bool): raise TypeError("Ifの条件はbool値であるべきです")
        if cond_val: val, deriv = evaluate(node.true_branch); return val, EIf(node, val, [cond_deriv, deriv], "E-IfT")
        else: val, deriv = evaluate(node.false_branch); return val, EIf(node, val, [cond_deriv, deriv], "E-IfF")
    if isinstance(node, BinaryOp):
        val1, deriv1 = evaluate(node.left); val2, deriv2 = evaluate(node.right)
        if not (isinstance(val1, int) and isinstance(val2, int)): raise TypeError(f"'{node.op}'演算子は整数にのみ適用できます")
        if isinstance(node, Plus): result, b_deriv, name = val1 + val2, BPlus(val1, val2, val1 + val2), "E-Plus"
        elif isinstance(node, Minus): result, b_deriv, name = val1 - val2, BMinus(val1, val2, val1 - val2), "E-Minus"
        elif isinstance(node, Times): result, b_deriv, name = val1 * val2, BTimes(val1, val2, val1 * val2), "E-Times"
        elif isinstance(node, LessThan): result, b_deriv, name = val1 < val2, BLt(val1, val2, val1 < val2), "E-Lt"
        else: raise TypeError("不明な二項演算子です")
        return result, EBinOp(node, result, [deriv1, deriv2, b_deriv], name)
    raise TypeError(f"不明な式の型です: {type(node)}")
# -------------------------------------------------------------------
# 5. メイン実行部
# -------------------------------------------------------------------
def main():
    print("導出したい式を入力してください。終了するにはCtrl+Dを押してください。")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            expr_str = line.split(" evalto ", 1)[0]
            try:
                ast = run_parser_on_text(expr_str)
                _, derivation_tree = evaluate(ast)
                print(derivation_tree.format())
                print("-" * 20)
            except (SyntaxError, TypeError, ValueError) as e: print(f"エラー: {e}", file=sys.stderr)
    except KeyboardInterrupt: print("\n終了します。")

if __name__ == "__main__":
    main()