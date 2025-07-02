# eval_nameless_ml3.py
import sys
import re
from dataclasses import dataclass
from typing import List, Union, Tuple

# -------------------------------------------------------------------
# 1. 名前付きの式の構造 (入力言語のAST)
# -------------------------------------------------------------------
class Expression:
    def to_string(self) -> str: return str(self)
    def get_precedence(self): return 99
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

precedence = {'App': 4, 'Times': 3, 'Plus': 2, 'Minus': 2, 'LessThan': 1, 'If': 0, 'Let': 0, 'LetRec': 0, 'Fun': 0}

@dataclass
class BinaryOp(Expression):
    op: str; left: Expression; right: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self) -> str:
        my_prec = self.get_precedence()
        left_str = self.left.to_string()
        if self.left.get_precedence() < my_prec: left_str = f"({left_str})"
        right_str = self.right.to_string()
        if self.right.get_precedence() <= my_prec: right_str = f"({right_str})"
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
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
@dataclass
class Let(Expression):
    var: str; bound_expr: Expression; body_expr: Expression
    def __str__(self): return f"let {self.var} = {self.bound_expr} in {self.body_expr}"
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
@dataclass
class Fun(Expression):
    arg_name: str; body: Expression
    def __str__(self): return f"fun {self.arg_name} -> {self.body}"
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
@dataclass
class App(Expression):
    func_expr: Expression; arg_expr: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self) -> str:
        left_str = self.func_expr.to_string()
        if self.func_expr.get_precedence() < self.get_precedence(): left_str = f"({left_str})"
        right_str = self.arg_expr.to_string()
        if self.arg_expr.get_precedence() <= self.get_precedence(): right_str = f"({right_str})"
        return f"{left_str} {right_str}"
@dataclass
class LetRec(Expression):
    func_name: str; arg_name: str; func_body: Expression; let_body: Expression
    def __str__(self): return f"let rec {self.func_name} = fun {self.arg_name} -> {self.func_body} in {self.let_body}"
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)

# -------------------------------------------------------------------
# 2. 名無しの式の構造 (AST)
# -------------------------------------------------------------------
class DBExpression:
    def __str__(self) -> str: raise NotImplementedError
    def get_precedence(self): return 99
@dataclass
class DBIntLiteral(DBExpression):
    value: int
    def __str__(self): return str(self.value)
@dataclass
class DBBoolLiteral(DBExpression):
    value: bool
    def __str__(self): return str(self.value).lower()
@dataclass
class DBIndex(DBExpression):
    index: int
    def __str__(self): return f"#{self.index}"
@dataclass
class DBBinaryOp(DBExpression):
    op: str; left: DBExpression; right: DBExpression
    def get_precedence(self): return {'<': 1, '+': 2, '-': 2, '*': 3}.get(self.op, 0)
    def __str__(self):
        my_prec = self.get_precedence()
        left_str = str(self.left); right_str = str(self.right)
        if self.left.get_precedence() < my_prec: left_str = f"({left_str})"
        if self.right.get_precedence() <= my_prec: right_str = f"({right_str})"
        return f"{left_str} {self.op} {right_str}"
@dataclass
class DBIf(DBExpression):
    cond: DBExpression; true_branch: DBExpression; false_branch: DBExpression
    def get_precedence(self): return 0
    def __str__(self): return f"if {self.cond} then {self.true_branch} else {self.false_branch}"
@dataclass
class DBLet(DBExpression):
    bound_expr: DBExpression; body_expr: DBExpression
    def get_precedence(self): return 0
    def __str__(self): return f"let . = {self.bound_expr} in {self.body_expr}"
@dataclass
class DBFun(DBExpression):
    body: DBExpression
    def get_precedence(self): return 0
    def __str__(self): return f"fun . -> {self.body}"
@dataclass
class DBApp(DBExpression):
    func_expr: DBExpression; arg_expr: DBExpression
    def get_precedence(self): return 4
    def __str__(self):
        left_str = str(self.func_expr)
        if self.func_expr.get_precedence() < self.get_precedence(): left_str = f"({left_str})"
        right_str = str(self.arg_expr)
        if self.arg_expr.get_precedence() <= self.get_precedence(): right_str = f"({right_str})"
        return f"{left_str} {right_str}"
@dataclass
class DBLetRec(DBExpression):
    func_body: DBExpression; let_body: DBExpression
    def get_precedence(self): return 0
    def __str__(self): return f"let rec . = fun . -> {self.func_body} in {self.let_body}"

# -------------------------------------------------------------------
# 3. 値の定義 (DBValue)
# -------------------------------------------------------------------
class DBClosure:
    def __init__(self, val_env: 'List[Value]', func_body: 'DBFun'):
        self.val_env = val_env
        self.func_body = func_body
    def __str__(self):
        env_str = ", ".join(format_value(v) for v in self.val_env)
        return f"({env_str})[{self.func_body}]"
class DBRecClosure:
    def __init__(self, val_env: 'List[Value]', func_body: 'DBLetRec'):
        self.val_env = val_env
        self.func_body = func_body
    def __str__(self):
        env_str = ", ".join(format_value(v) for v in self.val_env)
        return f"({env_str})[rec . = fun . -> {self.func_body.func_body}]"

Value = Union[int, bool, DBClosure, DBRecClosure]
Token = Tuple[str, str]

def format_value(v: Value) -> str:
    if isinstance(v, bool): return str(v).lower()
    return str(v)

# -------------------------------------------------------------------
# 4. 導出規則 (Tr-Rules and E-Rules)
# -------------------------------------------------------------------
class Derivation:
    def format(self, i=0) -> str: raise NotImplementedError
class TrDerivation(Derivation): pass
class DBEDerivation(Derivation): pass
class BDerivation(Derivation): pass

def format_var_list(var_list: List[str]) -> str:
    if not var_list: return ""
    return ", ".join(var_list) + " "
def format_val_env(val_env: List[Value]) -> str:
    if not val_env: return ""
    return ", ".join(format_value(v) for v in val_env) + " "

# --- Translation Derivations ---
@dataclass
class TrInt(TrDerivation):
    var_list: List[str]; expr: IntLiteral
    def format(self, i=0): return f'{" "*i}{format_var_list(self.var_list)}|- {self.expr} ==> {self.expr} by Tr-Int {{}};'
@dataclass
class TrBool(TrDerivation):
    var_list: List[str]; expr: BoolLiteral
    def format(self, i=0): return f'{" "*i}{format_var_list(self.var_list)}|- {self.expr} ==> {self.expr} by Tr-Bool {{}};'
@dataclass
class TrVar1(TrDerivation):
    var_list: List[str]; name: str; index: int
    def format(self, i=0): return f'{" "*i}{format_var_list(self.var_list)}|- {self.name} ==> #{self.index} by Tr-Var1 {{}};'
@dataclass
class TrVar2(TrDerivation):
    var_list: List[str]; name: str; index: int; premise: TrDerivation
    def format(self, i=0) -> str:
        indent = " " * i; premise_str = self.premise.format(i + 1)
        return f"{indent}{format_var_list(self.var_list)}|- {self.name} ==> #{self.index} by Tr-Var2 {{\n{premise_str}\n{indent}}};"
@dataclass
class TrGeneric(TrDerivation):
    var_list: List[str]; expr: Expression; db_expr: DBExpression; rule_name: str; premises: List[TrDerivation]
    def format(self, i=0) -> str:
        indent = " " * i; premise_str = "\n".join(p.format(i + 1) for p in self.premises)
        return f"{indent}{format_var_list(self.var_list)}|- {self.expr} ==> {self.db_expr} by {self.rule_name} {{\n{premise_str}\n{indent}}};"

# --- Evaluation Derivations ---
@dataclass
class EInt(DBEDerivation):
    val_env: List[Value]; value: int
    def format(self, i=0): return f'{" "*i}{format_val_env(self.val_env)}|- {self.value} evalto {self.value} by E-Int {{}};'
@dataclass
class EBool(DBEDerivation):
    val_env: List[Value]; value: bool
    def format(self, i=0): return f'{" "*i}{format_val_env(self.val_env)}|- {format_value(self.value)} evalto {format_value(self.value)} by E-Bool {{}};'
@dataclass
class EVar(DBEDerivation):
    val_env: List[Value]; index: int; value: Value
    def format(self, i=0): return f'{" "*i}{format_val_env(self.val_env)}|- #{self.index} evalto {format_value(self.value)} by E-Var {{}};'
@dataclass
class EFun(DBEDerivation):
    val_env: List[Value]; expr: DBFun; value: DBClosure
    def format(self, i=0): return f'{" "*i}{format_val_env(self.val_env)}|- {self.expr} evalto {self.value} by E-Fun {{}};'
@dataclass
class EGeneric(DBEDerivation):
    val_env: List[Value]; expr: DBExpression; value: Value; rule_name: str; premises: List[Union[Derivation, 'BDerivation']]
    def format(self, i=0) -> str:
        indent = " " * i; premise_str = "\n".join(p.format(i + 1) for p in self.premises)
        val_str = format_value(self.value)
        return f"{indent}{format_val_env(self.val_env)}|- {self.expr} evalto {val_str} by {self.rule_name} {{\n{premise_str}\n{indent}}};"
@dataclass
class BPlus(BDerivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} plus {self.n2} is {self.result} by B-Plus {{}};'
@dataclass
class BMinus(BDerivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} minus {self.n2} is {self.result} by B-Minus {{}};'
@dataclass
class BTimes(BDerivation):
    n1: int; n2: int; result: int
    def format(self, i=0): return f'{" "*i}{self.n1} times {self.n2} is {self.result} by B-Times {{}};'
@dataclass
class BLt(BDerivation):
    n1: int; n2: int; result: bool
    def format(self, i=0): return f'{" "*i}{self.n1} less than {self.n2} is {format_value(self.result)} by B-Lt {{}};'

# -------------------------------------------------------------------
# 5. 変換エンジン (Translate)
# -------------------------------------------------------------------
def translate_var(var_list: List[str], name: str) -> Tuple[DBIndex, TrDerivation]:
    if not var_list: raise NameError(f"未定義の変数です: '{name}'")
    last_var = var_list[-1]; rest_list = var_list[:-1]
    if last_var == name: return DBIndex(1), TrVar1(var_list, name, 1)
    else:
        sub_db_expr, sub_deriv = translate_var(rest_list, name)
        new_index = sub_db_expr.index + 1
        return DBIndex(new_index), TrVar2(var_list, name, new_index, sub_deriv)
def translate(var_list: List[str], expr: Expression) -> Tuple[DBExpression, TrDerivation]:
    if isinstance(expr, IntLiteral): return DBIntLiteral(expr.value), TrInt(var_list, expr)
    if isinstance(expr, BoolLiteral): return DBBoolLiteral(expr.value), TrBool(var_list, expr)
    if isinstance(expr, Variable): return translate_var(var_list, expr.name)
    if isinstance(expr, BinaryOp):
        d1, p1 = translate(var_list, expr.left); d2, p2 = translate(var_list, expr.right)
        rule_map = {'+': 'Tr-Plus', '-': 'Tr-Minus', '*': 'Tr-Times', '<': 'Tr-Lt'}
        db_expr = DBBinaryOp(expr.op, d1, d2)
        return db_expr, TrGeneric(var_list, expr, db_expr, rule_map.get(expr.op, "Tr-Op"), [p1, p2])
    if isinstance(expr, If):
        d1, p1 = translate(var_list, expr.cond); d2, p2 = translate(var_list, expr.true_branch); d3, p3 = translate(var_list, expr.false_branch)
        db_expr = DBIf(d1, d2, d3)
        return db_expr, TrGeneric(var_list, expr, db_expr, "Tr-If", [p1, p2, p3])
    if isinstance(expr, Let):
        d1, p1 = translate(var_list, expr.bound_expr)
        d2, p2 = translate(var_list + [expr.var], expr.body_expr)
        db_expr = DBLet(d1, d2)
        return db_expr, TrGeneric(var_list, expr, db_expr, "Tr-Let", [p1, p2])
    if isinstance(expr, Fun):
        d, p = translate(var_list + [expr.arg_name], expr.body)
        db_expr = DBFun(d)
        return db_expr, TrGeneric(var_list, expr, db_expr, "Tr-Fun", [p])
    if isinstance(expr, App):
        d1, p1 = translate(var_list, expr.func_expr)
        d2, p2 = translate(var_list, expr.arg_expr)
        db_expr = DBApp(d1, d2)
        return db_expr, TrGeneric(var_list, expr, db_expr, "Tr-App", [p1, p2])
    if isinstance(expr, LetRec):
        d1, p1 = translate(var_list + [expr.func_name, expr.arg_name], expr.func_body)
        d2, p2 = translate(var_list + [expr.func_name], expr.let_body)
        db_expr = DBLetRec(d1, d2)
        return db_expr, TrGeneric(var_list, expr, db_expr, "Tr-LetRec", [p1, p2])
    raise TypeError(f"変換未対応の式の型です: {type(expr)}")

# -------------------------------------------------------------------
# 6. 評価エンジン (evaluate)
# -------------------------------------------------------------------
def evaluate(val_env: List[Value], db_expr: DBExpression) -> Tuple[Value, Derivation]:
    if isinstance(db_expr, DBIntLiteral): return db_expr.value, EInt(val_env, db_expr.value)
    if isinstance(db_expr, DBBoolLiteral): return db_expr.value, EBool(val_env, db_expr.value)
    if isinstance(db_expr, DBIndex):
        val = val_env[-(db_expr.index)]; return val, EVar(val_env, db_expr.index, val)
    if isinstance(db_expr, DBBinaryOp):
        v1, p1 = evaluate(val_env, db_expr.left); v2, p2 = evaluate(val_env, db_expr.right)
        if db_expr.op == '+': result = v1 + v2; b_deriv = BPlus(v1, v2, result); rule_name = "E-Plus"
        elif db_expr.op == '-': result = v1 - v2; b_deriv = BMinus(v1, v2, result); rule_name = "E-Minus"
        elif db_expr.op == '*': result = v1 * v2; b_deriv = BTimes(v1, v2, result); rule_name = "E-Times"
        elif db_expr.op == '<': result = v1 < v2; b_deriv = BLt(v1, v2, result); rule_name = "E-Lt"
        else: raise NotImplementedError(f"Unsupported op: {db_expr.op}")
        return result, EGeneric(val_env, db_expr, result, rule_name, [p1, p2, b_deriv])
    if isinstance(db_expr, DBIf):
        cond_val, p1 = evaluate(val_env, db_expr.cond)
        if not isinstance(cond_val, bool): raise TypeError("If condition must be a boolean")
        if cond_val:
            result, p2 = evaluate(val_env, db_expr.true_branch)
            return result, EGeneric(val_env, db_expr, result, "E-IfT", [p1, p2])
        else:
            result, p2 = evaluate(val_env, db_expr.false_branch)
            return result, EGeneric(val_env, db_expr, result, "E-IfF", [p1, p2])
    if isinstance(db_expr, DBLet):
        v1, p1 = evaluate(val_env, db_expr.bound_expr)
        v2, p2 = evaluate(val_env + [v1], db_expr.body_expr)
        return v2, EGeneric(val_env, db_expr, v2, "E-Let", [p1, p2])
    if isinstance(db_expr, DBFun):
        closure = DBClosure(val_env, db_expr)
        return closure, EFun(val_env, db_expr, closure)
    if isinstance(db_expr, DBApp):
        func_val, p1 = evaluate(val_env, db_expr.func_expr)
        arg_val, p2 = evaluate(val_env, db_expr.arg_expr)
        if isinstance(func_val, DBClosure):
            new_env = func_val.val_env + [arg_val]
            result, p3 = evaluate(new_env, func_val.func_body.body)
            return result, EGeneric(val_env, db_expr, result, "E-App", [p1, p2, p3])
        if isinstance(func_val, DBRecClosure):
            rec_body = func_val.func_body.func_body
            new_env = func_val.val_env + [func_val, arg_val]
            result, p3 = evaluate(new_env, rec_body)
            return result, EGeneric(val_env, db_expr, result, "E-AppRec", [p1, p2, p3])
        raise TypeError(f"関数でないものを適用しようとしました: {func_val}")
    if isinstance(db_expr, DBLetRec):
        rec_closure = DBRecClosure(val_env, db_expr)
        new_env = val_env + [rec_closure]
        result, p1 = evaluate(new_env, db_expr.let_body)
        return result, EGeneric(val_env, db_expr, result, "E-LetRec", [p1])
    raise TypeError(f"評価未対応の式の型です: {type(db_expr)}")

# -------------------------------------------------------------------
# 7. パーサー (Parser and DBParser)
# -------------------------------------------------------------------
class NamedParser:
    def __init__(self, tokens: List[Token]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        keywords = {'if', 'then', 'else', 'true', 'false', 'let', 'in', 'fun', 'rec'}
        token_spec = [('ARROW', r'->'), ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'), ('INT', r'-?\d+'), ('OP', r'[+*<()-=]'), ('SKIP', r'\s+'), ('MISMATCH', r'.')]
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_spec)
        tokens: List[Token] = []
        for mo in re.finditer(tok_regex, text):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            elif kind == 'MISMATCH': raise SyntaxError(f"認識できない文字です: '{value}'")
            if kind == 'ID' and value in keywords: kind = value.upper()
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
        if kind == 'FUN': return self.parse_fun()
        return self.parse_comparison()
    def parse_let(self) -> Expression:
        self.consume('let')
        if self.current_token() and self.current_token()[1] == 'rec': return self.parse_let_rec()
        _, var_name = self.current_token(); self.consume(var_name); self.consume('=')
        bound_expr = self.parse_expression(); self.consume('in'); body_expr = self.parse_expression()
        return Let(var_name, bound_expr, body_expr)
    def parse_let_rec(self) -> Expression:
        self.consume('rec'); _, func_name = self.current_token(); self.consume(func_name); self.consume('=')
        self.consume('fun'); _, arg_name = self.current_token(); self.consume(arg_name); self.consume('->')
        func_body = self.parse_expression(); self.consume('in'); let_body = self.parse_expression()
        return LetRec(func_name, arg_name, func_body, let_body)
    def parse_fun(self) -> Expression:
        self.consume('fun'); _, arg_name = self.current_token(); self.consume(arg_name); self.consume('->')
        body = self.parse_expression()
        return Fun(arg_name, body)
    def parse_if(self) -> Expression:
        self.consume('if'); cond = self.parse_expression(); self.consume('then')
        true_branch = self.parse_expression(); self.consume('else'); false_branch = self.parse_expression()
        return If(cond, true_branch, false_branch)
    def parse_comparison(self) -> Expression:
        node = self.parse_add_sub()
        if self.current_token() and self.current_token()[1] == '<': self.consume('<'); right_node = self.parse_add_sub(); node = LessThan(node, right_node)
        return node
    def parse_add_sub(self) -> Expression:
        node = self.parse_mul()
        while self.current_token() and self.current_token()[1] in ['+', '-']:
            _, op_val = self.current_token(); self.consume(op_val); right_node = self.parse_mul(); node = Plus(node, right_node) if op_val == '+' else Minus(node, right_node)
        return node
    def parse_mul(self) -> Expression:
        node = self.parse_app()
        while self.current_token() and self.current_token()[1] == '*': self.consume('*'); right_node = self.parse_app(); node = Times(node, right_node)
        return node
    def parse_app(self) -> Expression:
        node = self.parse_primary()
        while True:
            token = self.current_token()
            if token is None: break
            kind, val = token
            if kind in ['INT', 'ID', 'TRUE', 'FALSE'] or val == '(': node = App(node, self.parse_primary())
            else: break
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

class DBParser:
    def __init__(self, tokens: List[Token]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        keywords = {'if', 'then', 'else', 'true', 'false', 'let', 'in', 'fun', 'rec'}
        token_spec = [('ARROW', r'->'), ('DB_INDEX', r'#\d+'), ('KEYWORD', r'\b(if|then|else|true|false|let|in|fun|rec)\b'), ('INT', r'-?\d+'), ('OP', r'[+*<()-.=]'), ('SKIP', r'\s+'), ('MISMATCH', r'.')]
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_spec)
        tokens: List[Token] = []
        for mo in re.finditer(tok_regex, text):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            elif kind == 'MISMATCH': raise SyntaxError(f"認識できない文字です: '{value}'")
            if kind == 'KEYWORD': kind = value.upper()
            tokens.append((kind, value))
        return cls(tokens)
    def current_token(self) -> Token | None: return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self, expected_val=None):
        if expected_val and (self.current_token() is None or self.current_token()[1] != expected_val):
             raise SyntaxError(f"'{expected_val}'が予期されましたが、'{self.current_token()}'が見つかりました")
        self.pos += 1
    def parse_expression(self) -> DBExpression:
        kind, _ = self.current_token() if self.current_token() else (None, None)
        if kind == 'LET': return self.parse_let()
        if kind == 'IF': return self.parse_if()
        if kind == 'FUN': return self.parse_fun()
        return self.parse_comparison()
    def parse_let(self) -> DBExpression:
        self.consume('let')
        if self.current_token() and self.current_token()[1] == 'rec': return self.parse_let_rec()
        self.consume('.'); self.consume('='); bound_expr = self.parse_expression(); self.consume('in'); body_expr = self.parse_expression()
        return DBLet(bound_expr, body_expr)
    def parse_let_rec(self) -> DBExpression:
        self.consume('rec'); self.consume('.'); self.consume('='); self.consume('fun'); self.consume('.'); self.consume('->')
        func_body = self.parse_expression(); self.consume('in'); let_body = self.parse_expression()
        return DBLetRec(func_body, let_body)
    def parse_fun(self) -> DBExpression:
        self.consume('fun'); self.consume('.'); self.consume('->'); body = self.parse_expression()
        return DBFun(body)
    def parse_if(self) -> DBExpression:
        self.consume('if'); cond = self.parse_expression(); self.consume('then'); true_branch = self.parse_expression(); self.consume('else'); false_branch = self.parse_expression()
        return DBIf(cond, true_branch, false_branch)
    def parse_comparison(self) -> DBExpression:
        node = self.parse_add_sub()
        if self.current_token() and self.current_token()[1] == '<': self.consume('<'); right_node = self.parse_add_sub(); node = DBBinaryOp('<', node, right_node)
        return node
    def parse_add_sub(self) -> DBExpression:
        node = self.parse_mul()
        while self.current_token() and self.current_token()[1] in ['+', '-']:
            _, op_val = self.current_token(); self.consume(op_val); right_node = self.parse_mul(); node = DBBinaryOp(op_val, node, right_node)
        return node
    def parse_mul(self) -> DBExpression:
        node = self.parse_app()
        while self.current_token() and self.current_token()[1] == '*': self.consume('*'); right_node = self.parse_app(); node = DBBinaryOp('*', node, right_node)
        return node
    def parse_app(self) -> DBExpression:
        node = self.parse_primary()
        while True:
            token = self.current_token()
            if token is None: break
            kind, val = token
            if kind in ['INT', 'DB_INDEX', 'TRUE', 'FALSE'] or val == '(': node = DBApp(node, self.parse_primary())
            else: break
        return node
    def parse_primary(self) -> DBExpression:
        if self.current_token() is None: raise SyntaxError("式の途中で入力が終了しました")
        kind, val = self.current_token()
        if kind == 'INT': self.consume(val); return DBIntLiteral(int(val))
        if kind == 'DB_INDEX': self.consume(val); return DBIndex(int(val[1:]))
        if kind == 'TRUE': self.consume(val); return DBBoolLiteral(True)
        if kind == 'FALSE': self.consume(val); return DBBoolLiteral(False)
        if val == '(': self.consume('('); node = self.parse_expression(); self.consume(')'); return node
        raise SyntaxError(f"予期しないトークンです: '{val}'")

def run_parser(text: str) -> Expression:
    parser = NamedParser.from_text(text)
    node = parser.parse_expression()
    if parser.current_token() is not None: raise SyntaxError(f"解析完了後に余分なトークンがあります: '{parser.current_token()}'")
    return node
def run_db_parser(text: str) -> DBExpression:
    parser = DBParser.from_text(text)
    node = parser.parse_expression()
    if parser.current_token() is not None: raise SyntaxError(f"解析完了後に余分なトークンがあります: '{parser.current_token()}'")
    return node

# -------------------------------------------------------------------
# 8. メイン実行部
# -------------------------------------------------------------------
def parse_val_env(env_str: str) -> List[Value]:
    if not env_str.strip(): return []
    vals: List[Value] = []
    for val_str in env_str.split(','):
        val_str = val_str.strip()
        if val_str == 'true': vals.append(True)
        elif val_str == 'false': vals.append(False)
        else: vals.append(int(val_str))
    return vals

def main():
    print("変換(==>)または評価(evalto)したい式を入力してください。")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            
            try:
                if "==>" in line: # Translation Task
                    main_parts = line.split("==>", 1)
                    left_side = main_parts[0]
                    parts = left_side.split("|-", 1)
                    var_list_str, expr_str = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", left_side.strip())
                    var_list = [v.strip() for v in var_list_str.split(',') if v.strip()]
                    ast = run_parser(expr_str)
                    _, derivation_tree = translate(var_list, ast)
                    formatted_tree = derivation_tree.format(0)
                
                elif "evalto" in line: # Evaluation Task
                    main_parts = line.split("evalto", 1)
                    left_side = main_parts[0]
                    parts = left_side.split("|-", 1)
                    val_env_str, expr_str = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", left_side.strip())
                    val_env = parse_val_env(val_env_str)
                    db_ast = run_db_parser(expr_str)
                    _, derivation_tree = evaluate(val_env, db_ast)
                    formatted_tree = derivation_tree.format(0)
                
                else:
                    raise SyntaxError("入力には '==>' (変換) または 'evalto' を含めてください。")

                if formatted_tree.endswith(';'):
                    formatted_tree = formatted_tree[:-1]
                print(formatted_tree)
                print("-" * 20)

            except (SyntaxError, TypeError, ValueError, NameError, NotImplementedError) as e:
                print(f"エラー: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n終了します。")

if __name__ == "__main__":
    main()
